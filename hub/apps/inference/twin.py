"""The MuJoCo twin on the inference page: offscreen renders at a joint estimate (:class:`TwinRenderer`), the
live per-joint state the twin panel draws (:class:`TwinState`) and the CLOAK mask tint."""
from __future__ import annotations

import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from hub.apps._common import encode_jpeg
from hub.apps.inference.util import load_json
from digital_twin.hardware.state_estimator import (COUPLED_JOINTS, LINK_PARENT, ArmIMUEstimator,
                                                       joints_from_pitches)

TWIN_CAMERAS = ('overview_close', 'overview', 'side')
TWIN_STALE_S = 1.0                                 # twin panel: a driven joint older than this is not live


class TwinUnavailable(RuntimeError):
    pass


class NullEstimator:
    """The base estimator ``*.from_hardware_map`` wraps when only the IMU geometry is wanted."""
    joints = ('boom', 'bucket', 'stick')


class TwinRenderer:
    """Offscreen MuJoCo renders of the twin at a joint estimate, on one dedicated thread (GL
    contexts are thread-bound); joints missing from the estimate (slew, and the stick unless it is
    tracked) sit at their rest values. Lazily imports MuJoCo; raises TwinUnavailable without it."""

    W, H = 480, 360

    def __init__(self, profile: str = 'full4'):
        self.profile = profile
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference-twin')
        self._state = None
        self._error = None

    def _init(self):
        try:
            import mujoco

            from digital_twin.sim.kinematics import JOINTS, load_model, rest_q
        except Exception as ex:
            raise TwinUnavailable(f'MuJoCo twin unavailable in {sys.executable}: {ex}') from ex
        m = load_model(self.profile)
        d = mujoco.MjData(m)
        qadr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS}
        self._rest = dict(zip(JOINTS, map(float, rest_q())))
        self._state = (mujoco, m, d, qadr, mujoco.Renderer(m, height=self.H, width=self.W))

    def render(self, q: dict, lines=(), camera: str = 'overview_close') -> bytes:
        return self._pool.submit(self._render, dict(q), list(lines), camera).result(timeout=30)

    def _render(self, q, lines, camera):
        if self._error:
            raise TwinUnavailable(self._error)
        if self._state is None:
            try:
                self._init()
            except TwinUnavailable as ex:
                self._error = str(ex)
                raise
        mujoco, m, d, qadr, renderer = self._state
        from digital_twin.sim.viz import make_camera
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        for n, a in qadr.items():
            d.qpos[a] = float(q[n]) if q.get(n) is not None else self._rest[n]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=make_camera(m, camera, lookat=(0.12, -0.02, 0.24), distance=1.2))
        frame = np.ascontiguousarray(renderer.render())
        if lines:
            try:
                from digital_twin.sim.viz import hud
                frame = hud(frame, lines, warn_token='!')
            except Exception:
                pass
        return encode_jpeg(frame, 85)

    def close(self):
        def _close():
            if self._state is not None:
                self._state[4].close()
                self._state = None
        try:
            self._pool.submit(_close).result(timeout=5)
        except Exception:
            pass
        self._pool.shutdown(wait=False)


def tint_mask(img, mask):
    """Tint the robot pixels whose tokens CLOAK drops (magenta)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return img
    img = np.array(img, copy=True)
    img[m] = (0.45 * img[m] + 0.55 * np.array([255, 0, 200])).astype(np.uint8)
    return img


class TwinState:
    """The joint state the twin panel draws, per joint with where it comes from and how old it is.

    Each joint of the twin (slew, boom, stick, bucket) is, in order of preference: ``imu`` (the
    boom / stick / bucket link IMUs of the Pi's ``imus`` through state_estimator.ArmIMUEstimator and
    configs/hardware_map.json ``calibration.arm_imus``; a joint needs its link and its parent link
    fresh), ``locked`` (hardware_map ``joints`` / ``homing_only_joints`` status ``locked``: at its
    ``locked_joints`` value), ``dead_reckoned`` (the job's estimate, ageing once no job updates it),
    ``welded`` (in ``welded_joints``: where the control models weld it, ``locked_joints`` value), else
    ``assumed`` (the homed pose: no estimate at all). The slew is ``locked`` at 0 = facing forward: the
    base is never driven and the slew is not estimated (the policy runner mocks it, state_estimator.SlewMock), else
    ``assumed`` (the homed pose: no estimate at all). Live while every joint that is not locked /
    welded is younger than ``stale_s``. The hardware map is re-read when it changes (that restarts the
    IMU filters).
    Boom and stick are shown in the true pose (hardware_map ``true_pose.offset_rad``: the boom underside
    47 deg and the stick flat at the top stops), not the model convention the policies use.
    """

    def __init__(self, hardware_map, stale_s: float = TWIN_STALE_S):
        self.hardware_map = Path(hardware_map)
        self.stale_s = float(stale_s)
        self._lock = threading.Lock()
        self._cfg = None                                    # (mtime, hw, ArmIMUEstimator | None, why not)
        self._imus = {}                                     # name -> (live, hz, t_sample monotonic)
        self._tilt = 0.0                                    # base tilt from the last live homing (rereference_boom)

    def _config(self):
        try:
            mt = self.hardware_map.stat().st_mtime
        except OSError:
            mt = None
        if self._cfg is None or self._cfg[0] != mt:
            hw = load_json(self.hardware_map, {}) or {}
            est, why = None, None
            try:
                est = ArmIMUEstimator.from_hardware_map(NullEstimator(), hw)
                est.offset += self._tilt                    # keep the homed re-anchoring across map reloads
            except Exception as ex:
                why = str(ex)
            self._cfg = (mt, hw, est, why)
        return self._cfg

    def rereference_boom(self, boom_rad: float, t: float | None = None) -> float | None:
        """The boom is on its homed stop at ``boom_rad``: re-anchor the base tilt as the runner does after
        homing (ArmIMUEstimator.rereference_boom), so the panel reads the stop angle. Returns the shift [rad]
        or None without a fresh boom reading."""
        with self._lock:
            est = self._config()[2]
            if est is None:
                return None
            d = est.rereference_boom(boom_rad, time.monotonic() if t is None else t)
            if d is not None:
                self._tilt += d
            return d

    def update(self, imus: dict, t_recv: float):
        """One Pi status ``imus`` block ({name: {live, hz, age_s, sample, accel_mean}}) received at
        ``t_recv`` (monotonic); a sample's time is ``t_recv - age_s``, as PiSensorSource.imus."""
        with self._lock:
            est = self._config()[2]
            readings = {}
            for name, e in (imus or {}).items():
                if not isinstance(e, dict):
                    continue
                s = e.get('sample') or {}
                ts = t_recv - float(e.get('age_s') or 0.0)
                self._imus[name] = (bool(e.get('live')), e.get('hz'), ts)
                if e.get('live') and s.get('accel') is not None:
                    readings[name] = (np.asarray(e.get('accel_mean') or s['accel'], float),
                                      np.asarray(s.get('gyro') or [0, 0, 0], float), ts)
            if est is not None and readings:
                est.update(readings, t_recv)

    def link_pitch(self, link: str, now: float | None = None) -> float | None:
        """``link``'s absolute pitch [rad] from the arm IMU calibration (fresh readings only), else None."""
        with self._lock:
            est = self._config()[2]
            return None if est is None else est.fresh_pitches(time.monotonic() if now is None else now).get(link)

    def zero_slew(self) -> float:
        """The slew is locked at 0 (facing forward): nothing to re-zero."""
        return 0.0

    def snapshot(self, job_q=None, job_t=None, job_source=None, now=None) -> dict:
        """``job_q`` / ``job_t`` / ``job_source``: the job's estimate {joint: rad}, when it was set
        (monotonic) and its label (Inference.q_est / q_est_t / q_source)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            _mt, hw, est, why = self._config()
            measured, ages = {}, {}
            if est is not None:
                measured = joints_from_pitches(est.fresh_pitches(now))
                for n in measured:
                    links = [n] + ([LINK_PARENT[n]] if LINK_PARENT[n] else [])
                    ages[n] = now - min(est.t_imu[k] for k in links)
            imus = {n: {'live': live, 'hz': hz, 'age_s': now - ts} for n, (live, hz, ts) in self._imus.items()}
        fixed = {n: float(v.get('value_rad', 0.0)) for n, v in (hw.get('locked_joints') or {}).items()
                 if isinstance(v, dict)}
        status = {n: v.get('status', 'available' if part == 'joints' else None) for part in ('joints', 'homing_only_joints')
                  for n, v in (hw.get(part) or {}).items() if isinstance(v, dict)}
        welded = hw.get('welded_joints') or {}
        home = {n: float(h['home_value_rad']) for n, h in ((hw.get('homing') or {}).get('joints') or {}).items()
                if isinstance(h, dict) and 'home_value_rad' in h}
        joints, driven = {}, []
        true_off = {n: float(v) for n, v in ((hw.get('true_pose') or {}).get('offset_rad') or {}).items()}
        for n in COUPLED_JOINTS:
            if n in measured:
                q, src, age = measured[n], 'imu', ages[n]
            elif n == 'slew':                               # locked: never driven, never estimated
                q, src, age = 0.0, 'locked', None
            elif status.get(n) == 'locked':
                q, src, age = fixed.get(n, home.get(n, 0.0)), 'locked', None
            elif job_q and job_q.get(n) is not None:
                q, src, age = float(job_q[n]), 'dead_reckoned', None if job_t is None else now - job_t
            elif n in welded:
                q, src, age = fixed.get(n, 0.0), 'welded', None
            else:
                q, src, age = home.get(n, 0.0), 'assumed', None
            if src not in ('locked', 'welded') and not (n == 'slew' and src == 'assumed'):   # unmeasured, undriven base
                driven.append(age)
            q += true_off.get(n, 0.0)                    # model convention -> true pose (display only)
            joints[n] = {'rad': q, 'deg': math.degrees(q), 'source': src, 'age_s': age}
        disp = (hw.get('true_pose') or {}).get('display_deg') or {}
        for n, key in (('boom', 'boom_underside_from_horizontal'), ('stick', 'stick_from_horizontal')):
            d_ = disp.get(key)
            if isinstance(d_, dict) and all(k in joints for k in d_.get('of', [])):
                joints[n]['deg'] = sum(math.degrees(joints[k]['rad']) for k in d_['of']) + float(d_['add_deg'])
                joints[n]['deg_meaning'] = key.replace('_', ' ')    # 'rad' stays the twin's joint value
        live = all(a is not None and a <= self.stale_s for a in driven)
        age = None if any(a is None for a in driven) else max(driven, default=0.0)
        return {'joints': joints, 'live': live, 'age_s': age, 'stale_s': self.stale_s, 'job_source': job_source,
                'convention': 'true pose (hardware_map true_pose)' if true_off else 'model',
                'arm_imus': {'calibrated': est is not None, 'error': why, 'imus': imus,
                             'session': ((hw.get('calibration') or {}).get('arm_imus') or {}).get('session')}}
