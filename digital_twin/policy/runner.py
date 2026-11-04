"""Closed-loop inference runner: drive the REAL excavator from a 4-joint π0.5 checkpoint (the ``pi05_tabletop_excavator``
contract) served on Lambda by the CLOAK policy server (policy/training/cloak/server.py, OpenPI wire protocol).

The observation is always the excavator's Pi camera (dashboard ``/preview.jpg``, IMU from
``/api/status``) and the only backend is the GPIO bridge (hardware.gpio). Every run drives the machine (there is
no dry run): pulses are sent to the bridge, and the joint state comes from the arm IMUs (refused without them).

    python -m digital_twin.policy.runner --gateway-url ws://127.0.0.1:8040 \\
        --prompt "scoop the dirt" --max-seconds 5 --record DIR                    # policy run (moves the machine)
    python -m digital_twin.policy.runner --probe --gateway-url ws://127.0.0.1:8040   # metadata only, no MuJoCo
    python -m digital_twin.policy.runner --home-only                          # homing only (moves the machine)

(A MuJoCo executor for offline closed-loop tests lives in digital_twin/tests/test_policy_runner.py
only; it is not reachable from this CLI.)

A policy run does not home: the estimator starts at the standard pose (the machine is expected there, e.g.
after ``--home-only``). ``--home-only`` homes and exits: one pin at a time, stick up, boom up, bucket curled in,
each driven into its top hard stop (refused when the map has no stick pins)
(configs/hardware/hardware_map.json ``homing``; measured range / slower measured speed x 1.5 + 1 s); that pose is the
standard pose. ``--start-pose slew,boom,stick,bucket`` (degrees) moves there open loop before inference; the
default ``none`` starts inference at the standard pose. One log line per pulse keeps a UI updated.

Contract (digital_twin.policy.cloak.CloakPolicy, POLICY_HZ = 10 Hz, chunk <= 10): state and action are
[slew, boom, stick, bucket] in real joint space (the labels come from the calibrated twin, so no joint
transform); the policy gets the NATIVE camera frame (no padding / resizing / blanking) and, when its metadata
says ``needs_mask``, the robot-pixel mask ``SelfMasker.mask(q, frame.shape)`` at the estimate at t_obs (the
no-CLOAK baseline sees the robot). Boom, stick and bucket are pulsed (the pins of configs/hardware/hardware_map.json
``joints``); the slew is locked: it is never pulsed or estimated, and the policy's slew state is mocked as the
slew it last asked for (state_estimator.SlewMock, 0 at the start). State: boom / stick / bucket from the arm
IMUs (dead reckoning of the commanded pulses fills a joint whose IMUs went stale); ``summary.state_sources`` records which. Joints
whose pin direction is unverified are listed in a start-of-run WARNING and ``summary.unverified_signs``
(never blocked).

Loop, per cycle (UMI-style latency handling):
  1. grab the newest Pi frame, timestamped on the runner clock (receive time - camera age);
  2. build the state (estimate at t_obs, mocked slew) and, if needed, the robot mask;
  3. ``CloakPolicy.infer`` (hard timeout ``--infer-timeout-s``) with the frame, mask, state and prompt;
  4. action k (k = 1..chunk) targets ``t_obs + k/hz``; actions already past when inference returns
     (+ ``--exec-latency-s``) are skipped (the last one is kept if all are);
  5. targets are clipped, rate-limited by the measured speeds and compiled to sequential pulses
     with digital_twin.hardware.commands.compile_schedule and sent;
  6. the estimator integrates the same pulses.

Joint state with the arm IMUs (always used; ``python -m digital_twin.hardware.calibrate --live`` writes
``calibration.arm_imus``): the boom, stick and bucket IMUs measure every joint
(digital_twin.hardware.state_estimator.ArmIMUEstimator, the only IMU estimate). After homing the boom's known
stop re-anchors the base tilt (``--home-only``); a start pose is reached open loop; every cycle the
measured joints overwrite the dead-reckoned ones before the state is sent to the policy (dead reckoning only
fills a joint whose IMUs went stale). A run is refused without calibrated, live arm IMUs.

Safety: a policy run needs ``--max-seconds``. The policy loop sends only the
pins of boom, stick and bucket; drive pins are never sent (they are not in the pin map).
``PolicyRunner.stop()`` / Ctrl-C stop cooperatively (checked between pulses, every 50 ms while waiting, around
infer()); a second Ctrl-C interrupts hard. A watchdog releases all pins if the loop stalls > 2 s; every exit
path releases all pins.

The CLOAK client is imported lazily, so the unit tests (a fake policy injected into the runner) run without a
policy server, and ``--probe`` runs without MuJoCo.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from policy.configs.schema import POLICY_HZ, TASKS
from digital_twin.sim import pins
from digital_twin.sim.joint_config import real_joint_config

from ..hardware import gpio
from ..hardware.excavator import (HOMING_JOINTS, STALL_S, BridgeExecutor, RunnerError, Watchdog, bridge_target,
                               check_pulse_pins, home_values, homing_pulses, load_hw, sign_verified, stick_blocked)
from ..hardware.pi import DEFAULT_PI_HOST, DEFAULT_TOKEN_FILE, PiSensorSource, link_kind
from ..hardware.state_estimator import (ARM_LINKS, FULL4_JOINTS, ArmIMUEstimator, DeadReckoningEstimator, SlewMock,
                                       load_joint_config)

# The runner's only profile: the 4-joint CLOAK π0.5 contract (digital_twin.policy.cloak), native frame + robot
# mask, real-space radians, POLICY_HZ x 10 actions. Checkpoints in configs/hub/local_checkpoints.json name it.
PROFILE = 'pi05_tabletop_excavator'
PROFILES = {PROFILE: {'state': 'observation/excavator/joints', 'action': 'action/excavator/joint_target',
                      'state_dim': 4, 'action_dim': 4, 'joints': FULL4_JOINTS, 'backend': 'cloak',
                      'policy_space': 'calibrated'}}
DEFAULT_PROMPT = TASKS['dirt']
INFER_TIMEOUT_S = 5.0


def known_prompts() -> set[str]:
    """The prompts the policies are trained with (policy.configs.schema.TASKS)."""
    return set(TASKS.values())


# ------------------------------------------------------------------------ safety
def check_run_allowed(args) -> None:
    """A policy run needs an explicit --max-seconds bound; the arm IMUs are checked in setup
    (PolicyRunner._setup_arm_imus). No other calibration-state gates (user instruction): pin signs, ranges, speeds
    and the IMU mount never block a run."""
    if not args.home_only and args.max_seconds is None:
        raise RunnerError('run refused: a policy run needs an explicit --max-seconds bound')


def validate_prompt(prompt: str, log=print):
    known = known_prompts()
    if known and prompt not in known:
        log(f'WARNING: prompt {prompt!r} is not one of the training prompts {sorted(known)}')


# ------------------------------------------------------------------------ recording
class Recorder:
    def __init__(self, directory):
        self.dir = Path(directory); (self.dir / 'frames').mkdir(parents=True, exist_ok=True)
        self.steps, self.pulses, self.frames, self.events = [], [], [], []

    def frame(self, step, img):
        from PIL import Image
        Image.fromarray(img).save(self.dir / 'frames' / f'{step:05d}.jpg', quality=92)
        self.frames.append(img)

    def close(self, summary: dict):
        if self.steps:
            keys = sorted({k for s in self.steps for k in s}, key=lambda k: (k != 'step', k))
            with open(self.dir / 'steps.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
                for s in self.steps:
                    w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()})
        cols = ['phase', 'step', 't_abs', 't_start_s', 'joint', 'direction', 'duration_ms', 'gpio', 'label']
        with open(self.dir / 'pulses.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader()
            for p in self.pulses:
                w.writerow(p)
        (self.dir / 'run.json').write_text(json.dumps({**summary, 'events': self.events}, indent=2, default=str))
        if self.frames:
            try:
                import imageio.v2 as imageio
                with imageio.get_writer(self.dir / 'observations.mp4', fps=POLICY_HZ, macro_block_size=16) as wr:
                    for fr in self.frames:
                        wr.append_data(fr)
            except Exception:
                try:
                    import cv2
                    wr = cv2.VideoWriter(str(self.dir / 'observations.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), POLICY_HZ,
                                         (self.frames[0].shape[1], self.frames[0].shape[0]))
                    for fr in self.frames:
                        wr.write(np.ascontiguousarray(fr[..., ::-1]))
                    wr.release()
                except Exception as ex:          # frames/ JPEGs remain
                    self.events.append({'mp4_error': str(ex)})


# ------------------------------------------------------------------------ runner
def rate_limit_targets(q0, Q, dts, cfg, joints=FULL4_JOINTS):
    """Clip targets to the joint ranges and to the calibrated speeds between consecutive times."""
    lo = np.array([cfg[n]['lo'] for n in joints]); hi = np.array([cfg[n]['hi'] for n in joints])
    vp = np.array([cfg[n]['v_pos'] for n in joints]); vn = np.array([cfg[n]['v_neg'] for n in joints])
    out, prev = [], np.asarray(q0, float)
    for q, dt in zip(np.asarray(Q, float), dts):
        q = np.clip(q, lo, hi)
        q = np.clip(q, prev - vn * dt, prev + vp * dt)
        out.append(q); prev = q
    return np.array(out)


def slew_config(hw: dict) -> tuple[dict, str]:
    """Slew {lo, hi, v_pos, v_neg}: range from configs/hardware/robot.json, speeds per direction from the slew calibration
    (hardware_map ``slew.yaw_deg_s``: left = +slew, right = -slew), else the robot.json prior."""
    c = load_joint_config(joints=('slew',))['slew']
    y = (hw.get('slew') or {}).get('yaw_deg_s') or {}
    meas = {k: abs(float(y[w])) for k, w in (('v_pos', 'left'), ('v_neg', 'right')) if y.get(w)}
    if not meas:
        return c, 'speed: prior (robot.json)'
    for k, other in (('v_pos', 'v_neg'), ('v_neg', 'v_pos')):
        c[k] = math.radians(meas.get(k, meas.get(other)))
    return c, 'speed: measured (hardware_map slew' + (')' if len(meas) == 2 else ', one direction)')


DEFAULT_ONSET_S = 0.15
PRIOR_ACK_S = 0.07          # p90 bridge->ESP32 'state fresh' round trip measured 2026-09-27 (60-72 ms)


def pulse_onsets(hw: dict, joints=('boom', 'stick', 'bucket')) -> tuple[dict, str]:
    """Motor onset delay per joint [s] from hardware_map: calibration.speeds.<j>.dead_time_s (press to motion,
    hardware.calibrate; positive values only), the slew's from ``slew.onset_s``; else DEFAULT_ONSET_S."""
    cal = (hw.get('calibration') or {}).get('speeds') or {}
    out, notes = {}, []
    for j in joints:
        src = (hw.get('slew') or {}).get('onset_s') if j == 'slew' else (cal.get(j) or {}).get('dead_time_s')
        dt = [v for v in (src or {}).values() if isinstance(v, (int, float)) and v > 0]
        out[j] = max(dt) if dt else DEFAULT_ONSET_S
        notes.append(f'{j} {"calibration" if dt else "default"}')
    return out, ', '.join(notes)


def latency_stats(samples) -> dict:
    x = np.asarray([v for v in samples], float)
    if not len(x):
        return {'n': 0}
    ema = x[0]
    for v in x[1:]:
        ema = 0.8 * ema + 0.2 * v
    return {'n': int(len(x)), 'p50': float(np.median(x)), 'p90': float(np.percentile(x, 90)), 'ema': float(ema),
            'max': float(x.max())}


def select_actions(t_obs: float, t_now: float, n_actions: int, hz: float, exec_latency_s: float = 0.0):
    """UMI latency rule: action k (0-based) targets t_obs + (k+1)/hz; keep those still in the future
    at t_now + exec_latency_s. If all are stale, keep the last one. Returns (indices, times, n_skipped)."""
    times = t_obs + (np.arange(n_actions) + 1) / hz
    keep = np.flatnonzero(times > t_now + exec_latency_s)
    if len(keep) == 0:
        keep = np.array([n_actions - 1])
    return keep, times[keep], int(n_actions - len(keep))


def schedule_times(times, t_now: float, hz: float) -> np.ndarray:
    """Target times relative to now, strictly increasing and in the future (a stale fallback
    action gets one step period)."""
    rel = np.asarray(times, float) - t_now
    out, prev = [], 0.0
    for r in rel:
        r = r if r > prev + 1e-3 else prev + 1.0 / hz
        out.append(r); prev = r
    return np.array(out)


class PolicyRunner:
    """Real-machine runner: Pi camera + bridge (every run moves the machine).

    ``stop()`` (thread-safe; the policy rollout app calls it) interrupts the run cooperatively: checked
    between pulses (and every 50 ms while a pulse is being waited out), around ``infer()`` and
    between steps; all pins are released on every exit path. Tests inject ``executor`` / ``camera``
    (a MuJoCo executor with ``virtual_clock = True`` lives in the test suite only)."""

    def __init__(self, args, policy=None, executor=None, camera=None, estimator=None, hw=None, log=print, sensors=None):
        self.args, self.log = args, log
        self.spec = PROFILES[args.profile]
        self.joints = tuple(self.spec['joints'])          # state / action order; boom, stick, bucket are pulsed
        if policy is None and not args.home_only:          # the runner connects to the policy server itself
            if args.hz != POLICY_HZ or not 1 <= args.chunk <= 10:
                raise RunnerError(f'CLOAK π0.5 requires {POLICY_HZ} Hz and chunk 1..10')
        self.hw = hw or load_hw(args.hardware_map)
        # The policy is trained in the calibrated twin's joint space = real space: measured ranges anchored at the
        # homing stops, measured speeds (digital_twin.sim.joint_config.real_joint_config), priors where unmeasured.
        self.policy_space = self.spec['policy_space']
        fallbacks = []
        self.cfg, self.cfg_sources = real_joint_config(self.hw, load_joint_config(), fallbacks.append)
        if fallbacks:
            log(f'calibration: {len(fallbacks)} value(s) missing, using priors / the other measured direction')
        self.slew_cfg_source = None
        if 'slew' in self.joints:
            self.cfg['slew'], self.slew_cfg_source = slew_config(self.hw)
        self.policy, self.executor, self.camera, self.estimator = policy, executor, camera, estimator
        self.sensors = sensors
        self.arm_est = None                # ArmIMUEstimator when the boom / stick / bucket IMUs measure the state
        self._arm_stale_logged = False
        self.stop_event = threading.Event()
        self.pulse_log = []                # every pulse sent, with phase and time
        self.recorder = Recorder(args.record) if args.record else None
        self.watchdog = None
        self.estimate_label = 'dead reckoning'     # 'arm IMUs (measured)' once _setup_arm_imus succeeds
        self.summary = {'policy_space': self.policy_space, 'profile': args.profile, 'prompt': args.prompt,
                        'joints': list(self.joints),
                        'hz': args.hz,
                        'chunk': args.chunk, 'steps': 0, 'estimate': self.estimate_label}
        self.external_hook = None          # callable(runner, t) -> None: external estimator (AprilTag/VIO) hook
        self.exec_lead_s = 0.0 if args.exec_latency_s == 'auto' else float(args.exec_latency_s)
        self.latency_info = {}
        self.masker = None                 # SelfMasker: the robot-pixel mask the CLOAK server asks for
        self.needs_mask = False            # the server wants the robot-pixel mask (metadata needs_mask)
        self.slew_mock = None              # SlewMock (slew locked, policy fed its own desired slew)
        self.state_sources = {}            # {joint: {source: steps}}

    @property
    def virtual(self) -> bool:
        return bool(getattr(self.executor, 'virtual_clock', False))

    def stop(self, reason: str = 'stop requested'):
        """Request a cooperative stop (Ctrl-C handler / dashboard). Pins are released on exit."""
        if not self.stop_event.is_set():
            self.log(f'STOP: {reason} - finishing the current pulse wait and releasing all pins')
        self.summary.setdefault('stopped', reason)
        self.stop_event.set()
        ab = getattr(self.executor, 'abort', None)
        if ab is not None:
            ab.set()

    def _check_stop(self):
        if self.stop_event.is_set():
            raise RunnerError('stop requested')

    def _bounded(self, fn, timeout: float, what: str, beat: bool = False):
        """Run ``fn`` in a worker thread; raise on timeout or stop (checked every 50 ms)."""
        box = {}

        def work():
            try:
                box['v'] = fn()
            except BaseException as ex:               # re-raised in the caller
                box['e'] = ex
        th = threading.Thread(target=work, daemon=True, name=what)
        th.start()
        deadline = time.monotonic() + timeout
        while th.is_alive():
            self._check_stop()
            if time.monotonic() > deadline:
                raise RunnerError(f'{what} timed out after {timeout:.1f} s')
            if beat:
                self._beat()
            th.join(0.05)
        if 'e' in box:
            raise box['e']
        return box.get('v')

    # ---- setup
    def setup(self):
        a = self.args
        check_run_allowed(a)
        self.log('joint config (real space): ' + '; '.join(f'{j}: {v}' for j, v in self.cfg_sources.items()))
        if self.executor is None:
            self.executor = BridgeExecutor(self.hw, bridge_target(a), dry_run=False, log=self.log, joints=self.joints)
        if getattr(self.executor, 'abort', None) is not None and self.stop_event.is_set():
            self.executor.abort.set()
        if self.estimator is None:
            self.estimator = DeadReckoningEstimator(joint_cfg=self.cfg, joints=self.joints)
        if self.camera is None and (not a.home_only or self.sensors is None):
            self.camera = PiSensorSource(a.pi_host, a.token_file, stop_event=self.stop_event, log=self.log)
        if isinstance(self.camera, PiSensorSource):
            self.sensors = self.camera
        if hasattr(self.sensors, 'start'):
            self.sensors.start()                           # frames / IMU flow during homing already
        if isinstance(self.sensors, PiSensorSource):
            self.summary['pi_host'] = self.sensors.base
            self.summary['pi_link'] = link_kind(self.sensors.base)
        self._setup_arm_imus()
        self._setup_state()
        if isinstance(self.executor, BridgeExecutor):
            self.executor.connect()
            self.watchdog = Watchdog(self._stall)
        if not a.home_only and self.policy is None:
            from .cloak import CloakPolicy
            self.policy = CloakPolicy(a.gateway_url, a.prompt, connect_timeout=a.connect_timeout_s,
                                      infer_timeout=a.infer_timeout_s)
        if self.policy is not None:
            self.log('connecting to policy server')
            md = self._bounded(self.policy.metadata, a.connect_timeout_s, 'policy connect')
            self.summary['server_metadata'] = md
            self.log(f'policy server: cameras {md["camera_names"]} action_keys {md["action_keys"]} '
                     f'horizon {md["action_horizon"]}')
            if list(md.get('joints') or self.joints) != list(self.joints) or md.get('action_dim', 4) != len(self.joints):
                raise RunnerError(f"policy server joints {md.get('joints')} / action_dim {md.get('action_dim')} "
                                  f'do not match the {a.profile} profile {list(self.joints)}')
            if md.get('policy_space', self.policy_space) != self.policy_space:
                raise RunnerError(f"policy server joint space {md.get('policy_space')!r} is not the runner's "
                                  f'{self.policy_space!r} (real) joint space')
            self.needs_mask = bool(md.get('needs_mask'))
            self.summary['cloak'] = {'mode': md.get('mode'), 'needs_mask': self.needs_mask,
                                     'policy_space': self.policy_space}
            self.log(f"CLOAK policy: mode {md.get('mode')}, "
                     + ('robot mask sent with every native frame' if self.needs_mask else
                        'no mask (the robot stays visible: nothing is blanked)'))
        if self.needs_mask and not a.home_only and self.masker is None:
            from digital_twin.sim.self_mask import SelfMasker
            self.masker = SelfMasker(config_path=getattr(a, 'self_mask_config', None))
        if self.needs_mask and self.masker is not None:
            self.log(self.masker.describe() if hasattr(self.masker, 'describe') else 'self-mask: robot mask')
            self.summary['self_mask'] = {'config': str(getattr(self.masker, 'config_path', '')),
                                         'mount': getattr(self.masker, 'mount', None),
                                         'measured': bool((getattr(self.masker, 'cfg', None) or {}).get('measured')),
                                         'use': 'mask sent to the policy (frame not blanked)'}

    def _setup_state(self):
        """The slew is locked and mocked (SlewMock); per-joint state sources; a WARNING for the pulsed joints
        with unverified pin direction."""
        self.slew_mock = SlewMock(self.joints, 0.0, self.cfg['slew']['lo'], self.cfg['slew']['hi'])
        arm = 'arm IMUs (ArmIMUEstimator), dead reckoning while stale' if self.arm_est is not None else None
        dr = 'dead reckoning (commanded pulses, calibrated speeds)'
        src = {'slew': "mocked: the policy's last desired slew (base locked, never pulsed)"}
        src.update({j: arm or dr for j in ('boom', 'stick', 'bucket')})
        self.summary['state_sources'] = {j: src[j] for j in self.joints}
        self.summary['state_source_steps'] = self.state_sources       # {joint: {imu | dead_reckoning | mocked: steps}}
        self.log('state sources: ' + '; '.join(f'{j}: {v}' for j, v in self.summary['state_sources'].items()))
        unv = [j for j in self.joints if j != 'slew' and not sign_verified(self.hw, j)]
        self.summary['unverified_signs'] = unv
        if unv:
            self.log(f'WARNING: pin direction not verified on hardware for {unv} (hardware_map sign_verified=false); '
                     'the run goes on - check that these joints move the way the policy commands '
                     '(python -m digital_twin.hardware.calibrate --live)')

    def _setup_arm_imus(self):
        """The boom / stick / bucket IMUs give the joint state; the run is refused when they are not calibrated
        or not live (no fallback to dead reckoning at the start)."""
        why = None
        if not (self.hw.get('calibration') or {}).get('arm_imus'):
            why = 'configs/hardware/hardware_map.json has no calibration.arm_imus (python -m digital_twin.hardware.calibrate --live)'
        elif self.sensors is None:
            why = 'no Pi dashboard sensors'
        elif not hasattr(self.sensors, 'imus'):
            why = 'the sensor source has no per-link IMUs'
        else:
            try:
                got, _ = self.sensors.imus()
                missing = [n for n in ARM_LINKS if n not in got]
                why = f'arm IMUs not live: {missing}' if missing else None
            except Exception as ex:
                why = f'arm IMU read failed ({type(ex).__name__}: {ex})'
        if why:
            raise RunnerError(f'run refused: the arm IMUs give the joint state, but {why}')
        self.arm_est = ArmIMUEstimator.from_hardware_map(self.estimator, self.hw)
        self.estimator = self.arm_est
        self.estimate_label = 'arm IMUs (measured)'
        self.summary['estimate'] = self.estimate_label
        self.summary['arm_imus'] = {'session': self.hw['calibration']['arm_imus'].get('session')}
        self.log('IMU: boom, stick and bucket IMUs measure the joint state; the policy gets the measured state')

    def arm_window(self, duration_s: float = 0.5) -> dict:
        """At-rest average of every arm IMU (hardware.calibrate.window on the runner clock)."""
        from ..hardware.calibrate import CalibrationError, window
        try:
            return window(_RunnerArmIO(self), duration_s)
        except CalibrationError as ex:
            raise RunnerError(str(ex)) from None

    def arm_settle(self, duration_s: float = 0.5) -> dict[str, float]:
        """Restart the arm filters from an at-rest window and write the measured joints into the base."""
        w = self.arm_window(duration_s)
        t = self.now()
        self.arm_est.set_static({n: w[n]['accel'] for n in ARM_LINKS}, t)
        return self.arm_est.apply_correction(t)

    def now(self) -> float:
        return self.executor.time if self.virtual else time.monotonic()

    def _clip_real(self, q):
        lo = np.array([self.cfg[j]['lo'] for j in self.joints]); hi = np.array([self.cfg[j]['hi'] for j in self.joints])
        return np.clip(q, lo, hi)

    def update_exec_lead(self, probe: bool = True) -> float:
        """--exec-latency-s auto: p90 bridge ack latency (state probes + trigger acks, last 50) plus the
        largest motor onset of the joints the loop drives. A number keeps that fixed value."""
        a = self.args
        if a.exec_latency_s != 'auto':
            self.exec_lead_s = float(a.exec_latency_s)
            return self.exec_lead_s
        onsets, src = pulse_onsets(self.hw, tuple(j for j in self.joints if j != 'slew'))   # the slew is never pulsed
        onset = max(onsets.values())
        ex = self.executor
        if isinstance(ex, BridgeExecutor):
            if probe:
                try:
                    ex.probe_latency(heartbeat=self._beat)            # the stall watchdog is armed
                except Exception as e:
                    self.log(f'bridge latency probe failed ({type(e).__name__}: {e})')
            st = latency_stats([v for _, v in ex.latency_samples[-50:]])
            ack = st.get('p90', PRIOR_ACK_S)
            if not st['n']:
                self.log('bridge latency: not measured (probe failed); '
                         f'using the prior {PRIOR_ACK_S * 1000:.0f} ms')
        else:                                             # test executor: its modelled start latency
            st = {'n': 0}; ack = float(getattr(ex, 'latency_s', 0.0))
        self.exec_lead_s = ack + onset
        self.latency_info = {'ack': st, 'ack_p90_s': ack, 'onset_s': onset, 'onset_source': src, 'lead_s': self.exec_lead_s}
        self.log(f"exec lead: p90 bridge ack {ack * 1000:.0f} ms (n={st['n']}"
                 + (f", p50 {st['p50'] * 1000:.0f} ms, EMA {st['ema'] * 1000:.0f} ms" if st['n'] else '')
                 + f") + motor onset {onset * 1000:.0f} ms ({src}) = {self.exec_lead_s * 1000:.0f} ms")
        return self.exec_lead_s

    def _beat(self):
        if self.watchdog:
            self.watchdog.beat()

    def _stall(self):
        self.log(f"WATCHDOG: loop stalled > {STALL_S:.0f} s in {getattr(self, '_where', '?')} - releasing all pins")
        try:
            self.executor.release_all()
        finally:
            self.stop_event.set()
            if hasattr(self.executor, 'abort'):
                self.executor.abort.set()

    def read_imu(self):
        """Feed the newest arm IMU readings to the estimator; returns the measured bucket pitch
        (boom + stick + bucket) or None."""
        if self.arm_est is None:
            return None
        try:
            got, _ = self.sensors.imus()
        except Exception as ex:
            self.log(f'IMU read failed: {ex}')
            return None
        self.arm_est.update(got, self.now())
        m = self.arm_est.measured(self.now())
        return sum(m[j] for j in ARM_LINKS) if all(j in m for j in ARM_LINKS) else None

    def execute(self, pulses, phase, step=None, gap_ms=None):
        gap = (self.hw.get('defaults', {}).get('gap_ms', 150) if self.args.gap_ms is None else self.args.gap_ms) \
            if gap_ms is None else gap_ms
        self._check_stop()
        res = self.executor.execute(pulses, gap, phase=phase, heartbeat=self._beat)
        if pulses:
            self._obs_after = time.monotonic() + float(getattr(self.args, 'obs_settle_s', 0.05))
        self._check_stop()
        self.estimator.update_from_pulses(self._effective(pulses, gap), res['t_start'])
        for p in self._effective(pulses, gap):
            row = {**p, 'phase': phase, 'step': step, 't_abs': round(res['t_start'] + p['t_start_s'], 4)}
            self.pulse_log.append(row)
            if self.recorder:
                self.recorder.pulses.append(row)
        return res

    @staticmethod
    def _effective(pulses, gap_ms):
        """Pulse start times as the executors run them (sequential, gap after each pulse)."""
        out, prev_end = [], -math.inf
        for p in pulses:
            s = max(p['t_start_s'], prev_end + gap_ms / 1000.0)
            out.append({**p, 't_start_s': s}); prev_end = s + p['duration_ms'] / 1000.0
        return out

    def raise_boom(self, seconds: float):
        """After a normal policy end: one boom-up pulse (the boom's positive pin) to lift the bucket
        clear. Not sent after an emergency stop, a stop request or an error."""
        pin = pins.pin(self.hw, 'boom', 1)
        ms = int(round(seconds * 1000))
        self.log(f'policy ended: raising the boom for {seconds:g} s (GPIO {pin})')
        self.execute([{'t_start_s': 0.0, 'joint': 'boom', 'direction': 1, 'duration_ms': ms, 'gpio': pin,
                       'label': 'end: boom up'}], 'end_raise')
        self.summary['end_raise_s'] = seconds

    # ---- homing
    def home(self) -> dict:
        a = self.args
        why = stick_blocked(self.hw)
        if why:                                      # the stick is always homed: no stick pins, no homing
            raise RunnerError(f'homing refused: {why} (the stick is homed with boom and bucket)')
        gap = self.hw.get('defaults', {}).get('gap_ms', 150) if a.gap_ms is None else a.gap_ms
        pulses = homing_pulses(self.hw, gap, self.cfg, compute=True)
        home = home_values(self.hw)
        for p in pulses:
            self.log(f"homing {p['joint']}: direction {p['direction']:+d} (GPIO {p['gpio']}, {p['label']}) for "
                     f"{p['duration_ms'] / 1000:.2f} s -> stop {home[p['joint']]:+.4f} rad "
                     f"({math.degrees(home[p['joint']]):+.1f} deg)")
        self.log(f'homing: {len(pulses)} pulses, about {sum(p["duration_ms"] for p in pulses) / 1000 + gap / 1000 * len(pulses):.1f} s')
        res = self.execute(pulses, 'homing')
        self.log('homing: done')
        self.estimator.reset(home)
        t = self.now()
        est = self.estimator.as_dict(t)
        report = {'pulses': pulses, 'home_values_rad': home, 'estimator_rad': est, 'joints_homed': list(HOMING_JOINTS),
                  'directions': {p['joint']: p['direction'] for p in pulses}, 'duration_s': res['t_end'] - res['t_start']}
        if hasattr(self.executor, 'qpos_by_name'):              # test executor (MuJoCo)
            self.executor.advance(0.5)
            q = self.executor.qpos_by_name()
            err = {n: q[n] - home[n] for n in home if n in q}
            report['sim_qpos_rad'] = q; report['sim_error_rad'] = err
            self.log('homing (sim): qpos error vs stops ' + ', '.join(f'{n} {e:+.2e} rad' for n, e in err.items()))
        self._home_imu_check(pulses, report)
        if self.arm_est is not None:
            self._arm_after_homing(home, report)
        self.log('standard pose: ' + ', '.join(f'{n} {v:+.4f} rad' for n, v in home.items()) + f' [{self.estimate_label}]')
        self.summary['homing'] = report
        if self.recorder:
            self.recorder.events.append({'homing': report})
        return report

    def _idle(self, dt: float):
        """Wait dt on the runner clock, checking for a stop every 50 ms."""
        if self.virtual:
            self.executor.advance(dt); return
        end = time.monotonic() + dt
        while time.monotonic() < end:
            self._check_stop(); self._beat(); time.sleep(min(0.05, max(0.0, end - time.monotonic())))

    def sample_imu(self, duration_s: float = 1.0, period_s: float = 0.05):
        """Average bucket IMU accel over ``duration_s`` -> (unit gravity vector in the IMU frame, raw mean accel [g],
        n samples) or None without an IMU."""
        if self.sensors is None:
            return None
        acc, t_end = [], self.now() + duration_s
        while True:
            try:
                got, _ = self.sensors.imu()
            except Exception as ex:
                self.log(f'IMU read failed: {ex}'); got = None
            if got is not None:
                acc.append(np.asarray(got[0], float))
            if self.now() >= t_end:
                break
            self._idle(period_s)
        if not acc:
            return None
        a = np.mean(acc, 0)
        return a / np.linalg.norm(a), a, len(acc)

    def _arm_after_homing(self, home: dict, report: dict):
        """Boom on its top stop: re-anchor the base tilt there, then measure every joint."""
        w = self.arm_window(0.5)
        t = self.now()
        self.arm_est.set_static({n: w[n]['accel'] for n in ARM_LINKS}, t)
        shift = self.arm_est.rereference_boom(home['boom'], t)
        m = self.arm_est.apply_correction(t)
        report['arm_imus_after_homing'] = {'measured_deg': {j: round(math.degrees(v), 2) for j, v in m.items()},
                                           'tilt_shift_deg': None if shift is None else round(math.degrees(shift), 2)}
        self.log('arm IMUs after homing: ' + ', '.join(f'{j} {math.degrees(v):+.1f} deg' for j, v in m.items())
                 + ('' if shift is None else f'; base tilt re-anchored by {math.degrees(shift):+.2f} deg'))
        for j in ('stick', 'bucket'):
            if j in m and abs(m[j] - home[j]) > math.radians(5):
                self.log(f'WARNING: {j} measured {math.degrees(m[j]):+.1f} deg after homing, its top stop is '
                         f'{math.degrees(home[j]):+.1f} deg' + (' (put the stick on its top stop by hand)' if j == 'stick' else ''))

    def _home_imu_check(self, pulses, report):
        """Log one averaged bucket IMU reading after homing (no check, no abort path - user instruction)."""
        if self.sensors is None:
            return
        try:
            r = self.sample_imu(0.5)
        except RunnerError:
            raise
        except Exception as ex:
            self.log(f'IMU after homing: read failed ({ex})'); return
        if r is None:
            self.log('IMU after homing: not live'); return
        g, acc, n = r
        report['imu_after_homing'] = {'accel_mean_g': acc.round(5).tolist(), 'n': n}
        self.log(f'IMU after homing (logged only): bucket gravity {np.round(g, 3).tolist()}')

    def move_to(self, q_target, phase='start_pose'):
        """Move the profile's joints to q_target through the policy pulse path (rate-limited, compiled)."""
        from ..hardware.commands import compile_schedule
        t = self.now()
        q0 = self.estimator.policy_state(t, self.joints)
        vp = np.array([self.cfg[n]['v_pos'] for n in self.joints]); vn = np.array([self.cfg[n]['v_neg'] for n in self.joints])
        if self.slew_mock is not None:                     # the slew is locked: never moved
            q_target = self.slew_mock.lock(q_target, q0)
        dq = np.asarray(q_target, float) - q0
        T = float(np.max(np.abs(dq) / np.where(dq >= 0, vp, vn))) + 1e-3
        sched = compile_schedule([0.0, T], np.vstack([q0, q_target]), v_pos=vp, v_neg=vn, hw_map=self.hw,
                                 min_pulse_ms=self.args.min_pulse_ms, gap_ms=self.args.gap_ms, joints=self.joints)
        check_pulse_pins(sched['pulses'], self.hw, 'policy', self.joints)
        self.log(f'{phase}: {np.degrees(q0).round(1).tolist()} -> {np.degrees(q_target).round(1).tolist()} deg '
                 f'({len(sched["pulses"])} pulses)')
        self.execute(sched['pulses'], phase)
        self.log(f'{phase}: done')
        return sched

    # ---- loop
    def step(self, k: int) -> dict:
        from ..hardware.commands import compile_schedule
        a = self.args
        self._beat()
        self._check_stop()
        w0 = time.perf_counter()
        self._where = f'step {k} camera grab'
        after = getattr(self, '_obs_after', None)
        try:
            raw, t_obs, meta = self.camera.grab(after=after)
        except TypeError:                                     # test cameras without the keyword
            raw, t_obs, meta = self.camera.grab()
        w_grab = time.perf_counter()
        self._where = f'step {k} state/mask'
        if abs(t_obs - self.now()) > 5.0:
            raise RunnerError(f'camera timestamp {t_obs:.3f} is not on the runner clock ({self.now():.3f}); '
                              'refusing to schedule against it')
        obs = np.ascontiguousarray(raw)     # the native frame as is (the client letterboxes frame and mask together)
        self._timing = {'grab_ms': (w_grab - w0) * 1000, 'fetch_wait_ms': meta.get('fetch_wait_ms'),
                        'frame_age_ms': meta.get('frame_age_ms'), 'mask_ms': 0.0}
        imu_pitch = self.read_imu()
        measured = {}
        if self.arm_est is not None:
            measured = self.arm_est.apply_correction(self.now())
            stale = [j for j in self.joints if j in ARM_LINKS and j not in measured]
            if stale and not self._arm_stale_logged:
                self.log(f'WARNING: arm IMUs stale for {stale}; dead reckoning fills them until they return')
            self._arm_stale_logged = bool(stale)
        if self.external_hook:
            self.external_hook(self, self.now())
        q_obs = self._clip_real(self.estimator.policy_state(t_obs, self.joints))      # real = policy joint space
        # the slew is locked, so the policy is fed back the slew it last asked for
        state = q_obs.copy() if self.slew_mock is None else self.slew_mock.state(q_obs)
        mask = None
        src = {j: ('mocked' if j == 'slew' else 'imu' if j in measured else 'dead_reckoning') for j in self.joints}
        for j, v in src.items():
            c = self.state_sources.setdefault(j, {}); c[v] = c.get(v, 0) + 1
        meta = {**meta, 'state_source': src}
        if self.needs_mask:                                  # robot pixels at the state sent with this frame
            m0 = time.perf_counter()
            mask = np.asarray(self.masker.mask(dict(zip(self.joints, map(float, q_obs))), obs.shape[:2]), bool)
            meta = {**meta, 'self_mask_frac': round(float(mask.mean()), 4),
                    'self_mask_ms': round((time.perf_counter() - m0) * 1000, 1)}
            self._timing['mask_ms'] = (time.perf_counter() - m0) * 1000
        self._beat()
        t_send = self.now()
        self._where = f'step {k} inference'
        # One slow round trip of the frame through the tunnel can pass the 2 s stall limit. No pulse is in flight
        # here (execute() returned after the last one ended and the ESP32 times every trigger itself), so keep
        # beating; --infer-timeout-s still bounds the call.
        actions, latency, server_ms = self._bounded(lambda: self.policy.infer(obs, state, mask), a.infer_timeout_s,
                                                    'inference', beat=True)
        if self.virtual:                                  # test executor: charge the latency to sim time
            self.executor.advance(getattr(self.executor, 'infer_latency_s', None) or latency)
        self._beat()
        if self.watchdog and self.watchdog.tripped.is_set():
            raise RunnerError('watchdog tripped during inference')
        t_now = self.now()
        n = min(a.chunk, len(actions))
        q_now = self.estimator.policy_state(t_now, self.joints)
        Q_pol = np.asarray(actions, float)[:n, :len(self.joints)].copy()
        w_c0 = time.perf_counter()
        Q_all = Q_pol.copy()
        keep, times, n_skipped = select_actions(t_obs, t_now, n, a.hz, self.exec_lead_s)
        slew_desired = None
        if self.slew_mock is not None:                    # remember the desired slew, then hold the real one
            slew_desired = self.slew_mock.command(Q_all[keep])
            Q_all = self.slew_mock.lock(Q_all, q_now)
        rel = schedule_times(times, t_now, a.hz)
        Q = rate_limit_targets(q_now, Q_all[keep], np.diff(np.r_[0.0, rel]), self.cfg, self.joints)
        sched = compile_schedule(np.r_[0.0, rel], np.vstack([q_now, Q]), windows=list(rel),
                                 v_pos=np.array([self.cfg[j]['v_pos'] for j in self.joints]),
                                 v_neg=np.array([self.cfg[j]['v_neg'] for j in self.joints]),
                                 hw_map=self.hw, min_pulse_ms=a.min_pulse_ms, gap_ms=a.gap_ms, joints=self.joints)
        check_pulse_pins(sched['pulses'], self.hw, 'policy', self.joints)
        w_c1 = time.perf_counter()
        self._where = f'step {k} execute'
        res = self.execute(sched['pulses'], 'policy', step=k)
        w_x = time.perf_counter()
        unsent = np.asarray(Q[-1], float) - np.asarray(sched['q_final_commanded'], float)   # below the minimum pulse
        t_next = max(res['t_end'], t_obs + 1.0 / a.hz)
        if t_next > self.now():
            self.executor.advance(t_next - self.now())
        timing = {**self._timing, 'inference_ms': latency * 1000, 'compile_ms': (w_c1 - w_c0) * 1000,
                  'execute_ms': (w_x - w_c1) * 1000, 'step_ms': (time.perf_counter() - w0) * 1000}
        timing = {k2: (None if v is None else round(float(v), 1)) for k2, v in timing.items()}
        rec = {'step': k, **{f't_{k2}': v for k2, v in timing.items()}, 't_obs': round(t_obs, 4), 't_send': round(t_send, 4), 't_actions': round(t_now, 4),
               'latency_ms': round(latency * 1000, 2), 'server_ms': server_ms, 'n_actions': int(len(actions)),
               'n_skipped': n_skipped, 'state': np.round(state, 6).tolist(),
               'state_real': np.round(q_obs, 6).tolist(), 'targets_policy': np.round(Q_pol[keep], 6).tolist(),
               'actions': np.round(actions[:max(n, 1)], 6).tolist(), 'targets': np.round(Q, 6).tolist(),
               'n_pulses': len(sched['pulses']), 'schedule_s': round(sched['duration_s'], 4),
               'unsent_deg': np.degrees(unsent).round(3).tolist(), 'exec_lead_ms': round(self.exec_lead_s * 1000, 1),
               **{f'est_{j}': round(v, 6) for j, v in self.estimator.as_dict(self.now()).items()},
               'imu_pitch_rad': None if imu_pitch is None else round(imu_pitch, 6),
               'imu_joints': {j: round(v, 6) for j, v in measured.items()},
               'camera_meta': {k2: v for k2, v in meta.items() if k2 != 'status'}}
        rec['estimate'] = self.estimate_label
        if slew_desired is not None:
            rec['slew_desired'] = round(slew_desired, 6)
        if hasattr(self.executor, 'qpos_by_name'):
            rec.update({f'sim_{j}': round(v, 6) for j, v in self.executor.qpos_by_name().items()})
        self.log(f"step {k:4d}: latency {rec['latency_ms']:7.1f} ms, skipped {n_skipped}/{n}, target policy "
                 f"{np.degrees(Q_pol[keep][-1]).round(1).tolist()} / sent {np.degrees(Q[-1]).round(1).tolist()} deg, "
                 f"state {np.degrees(state).round(1).tolist()} deg, "
                 f"{len(sched['pulses'])} pulses{'' if imu_pitch is None else f', IMU pitch {math.degrees(imu_pitch):.1f} deg'} "
                 f"[{self.estimate_label}]")
        if np.any(np.abs(unsent) > 1e-6):
            self.log(f"  unsent: {', '.join(f'{j} {math.degrees(c):+.2f} deg' for j, c in zip(self.joints, unsent))} "
                     'below the minimum pulse')
        self.log(f"  timing: fetch wait {timing['fetch_wait_ms']} ms, frame age {timing['frame_age_ms']} ms, "
                 f"grab {timing['grab_ms']} ms, mask {timing['mask_ms']} ms, inference {timing['inference_ms']} ms, "
                 f"compile {timing['compile_ms']} ms, execute {timing['execute_ms']} ms, step {timing['step_ms']} ms")
        if self.recorder:
            self.recorder.frame(k, obs[::2, ::2])                         # native frames: half size on disk
            self.recorder.steps.append(rec)
        return rec

    def run(self) -> dict:
        a = self.args
        try:
            self.setup()
            if self.watchdog:
                self.watchdog.arm()
            if a.home_only:
                self.home()
            else:
                self.log('no homing: the estimator assumes the standard pose')
                self.estimator.reset(home_values(self.hw))
            if self.slew_mock is not None:            # slew 0 = the base as it faces now (forward, set by the operator)
                self.slew_mock.reset(0.0)
            if a.home_only:
                return self.summary
            if a.start_pose != 'none':
                q = np.radians([float(x) for x in a.start_pose.split(',')])     # real = policy joint space
                if len(q) != len(self.joints):
                    raise RunnerError(f'--start-pose needs {len(self.joints)} angles ({",".join(self.joints)} deg)')
                self.summary['start_pose'] = {'deg': np.degrees(q).round(2).tolist()}
                self.move_to(q)
            self._bounded(self.policy.reset, a.infer_timeout_s, 'policy reset')
            self.update_exec_lead()
            t_start = self.now()
            limit = a.max_seconds
            k = 0
            while (self.now() - t_start < limit and (a.max_steps is None or k < a.max_steps)
                   and not self.stop_event.is_set()):
                if k and k % 30 == 0:
                    self.update_exec_lead()
                self.step(k)
                k += 1
                self.summary['steps'] = k
            self.summary['duration_s'] = self.now() - t_start
            if a.end_raise_s > 0 and not self.stop_event.is_set():
                self.raise_boom(a.end_raise_s)
            if isinstance(self.sensors, PiSensorSource):
                self.summary['pi_host'] = self.sensors.base
                self.summary['pi_link'] = link_kind(self.sensors.base)
                self.summary['pi_host_events'] = [{'host': h, 'reason': r} for _, h, r in self.sensors.host_events]
            self.summary['exec_lead'] = self.latency_info or {'lead_s': self.exec_lead_s, 'source': 'fixed'}
            return self.summary
        except KeyboardInterrupt:
            self.log('EMERGENCY STOP (Ctrl-C): releasing all pins')
            self.summary['stopped'] = 'keyboard_interrupt'
            self.stop_event.set()
            raise
        except Exception as ex:
            if self.stop_event.is_set() and not (self.watchdog and self.watchdog.tripped.is_set()):
                self.summary.setdefault('stopped', 'stop requested')
                self.log(f'stopped: {self.summary["stopped"]}')
                return self.summary
            self.log(f'ERROR: {type(ex).__name__}: {ex} - releasing all pins')
            self.summary['error'] = f'{type(ex).__name__}: {ex}'
            raise
        finally:
            if self.watchdog:
                self.watchdog.stop()
            if self.executor is not None:
                try:
                    self.executor.release_all()
                finally:
                    if hasattr(self.executor, 'close'):
                        self.executor.close()
            if self.policy is not None:
                self.policy.close()
            if self.masker is not None:
                self.masker.close()
            for src in {id(x): x for x in (self.camera, self.sensors) if x is not None}.values():
                if hasattr(src, 'close'):
                    src.close()
            if self.recorder:
                lat = [s['latency_ms'] for s in self.recorder.steps]
                if lat:
                    self.summary['latency_ms'] = {'median': float(np.median(lat)), 'max': float(np.max(lat))}
                self.recorder.close(self.summary)


class _RunnerArmIO:
    """hardware.calibrate rig interface on the runner: its sensors, clock, stop checks and pulse path."""

    def __init__(self, runner):
        self.r = runner

    def read(self) -> dict:
        return self.r.sensors.imus()[0]

    def now(self) -> float:
        return self.r.now()

    def sleep(self, s: float):
        self.r._idle(s)

    def pulse(self, joint, direction, ms):
        pin = pins.pin(self.r.hw, joint, direction)
        self.r.execute([{'t_start_s': 0.0, 'joint': joint, 'direction': int(direction), 'duration_ms': int(ms),
                         'gpio': pin, 'label': f'start pose {joint} {direction:+d}'}], 'start_pose')

    def operator(self, key, message) -> bool:
        return False


# ------------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--profile', choices=sorted(PROFILES), default=PROFILE,
                   help='policy contract (the only one: pi05_tabletop_excavator, as configs/hub/local_checkpoints.json names it)')
    p.add_argument('--gateway-url', help='CLOAK policy server on the Lambda box (ws://127.0.0.1:<port> through the SSH '
                                         'forward; see configs/hub/local_checkpoints.json)')
    p.add_argument('--prompt', default=DEFAULT_PROMPT, help=f'task prompt (trained: {sorted(known_prompts())})')
    p.add_argument('--pi-host', default=DEFAULT_PI_HOST,
                   help='excavator Pi dashboard host:port, or an ordered comma list (first that answers /api/status '
                        'is used; re-probed on repeated failures). Default: hosts.pi.urls in configs/hub/local.json')
    p.add_argument('--token-file', default=str(DEFAULT_TOKEN_FILE), help='Pi dashboard access token file')
    p.add_argument('--bridge-host', default=gpio.DEFAULT_HOST,
                   help='GPIO bridge: the hub host:port (its /api/gpio/ws websocket). In-process callers pass a '
                        'GpioBridge as the namespace attribute "bridge" instead')
    p.add_argument('--hz', type=float, default=POLICY_HZ,
                   help=f'action time base: action k targets t_obs + k/hz. Fixed by training ({POLICY_HZ} Hz); the loop itself '
                        'runs as fast as observation + inference + execution allow')
    p.add_argument('--chunk', type=int, default=8, help='actions considered per inference (stale ones are skipped)')
    p.add_argument('--max-seconds', type=float, default=None, help='loop time bound (required for a policy run)')
    p.add_argument('--end-raise-s', type=float, default=1.0,
                   help='after a normal end, pulse the boom up for this long (0 = off; never after a stop)')
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--record', metavar='DIR')
    p.add_argument('--home-only', action='store_true',
                   help='drive stick, boom and bucket into their top hard stops (the standard pose) and exit')
    p.add_argument('--start-pose', default='none',
                   help="after homing: 'none' (default: stay at the standard pose) or 'slew,boom,stick,bucket' in "
                        'degrees (real joint space; the slew stays locked)')
    p.add_argument('--exec-latency-s', type=lambda v: v if v == 'auto' else float(v), default='auto',
                   help="execution lead before an action counts as stale: 'auto' (default) = measured p90 bridge "
                        'ack latency + motor onset, re-measured every 30 steps; or seconds')
    p.add_argument('--infer-timeout-s', type=float, default=INFER_TIMEOUT_S, help='hard timeout per infer() / reset()')
    p.add_argument('--obs-settle-s', type=float, default=0.05,
                   help='after sending pulses, only use camera frames received this long after they end')
    p.add_argument('--connect-timeout-s', type=float, default=60.0, help='hard timeout for the first gateway connect')
    p.add_argument('--self-mask-config', default=None,
                   help='robot-mask config for servers that need the mask (default configs/hardware/self_mask.json)')
    p.add_argument('--gap-ms', type=int, default=None)
    p.add_argument('--min-pulse-ms', type=int, default=None)
    p.add_argument('--hardware-map', default=None)
    p.add_argument('--probe', action='store_true', help='connect, print the policy server metadata; no actions')
    return p


def main(argv=None) -> int:
    import signal
    args = build_parser().parse_args(argv)
    if args.probe:
        if not args.gateway_url:
            print('--probe needs --gateway-url', file=sys.stderr); return 2
        from .cloak import CloakPolicy
        pol = CloakPolicy(args.gateway_url, args.prompt)
        try:
            print(json.dumps(pol.metadata(), indent=2, default=str))
        except Exception as ex:
            print(f'PROBE FAILED: {ex}', file=sys.stderr)
            return 2
        finally:
            pol.close()
        return 0
    if not args.home_only:
        if not args.gateway_url:
            print('--gateway-url is required (or --probe / --home-only)', file=sys.stderr); return 2
        validate_prompt(args.prompt)
    try:
        runner = PolicyRunner(args)
    except RunnerError as ex:
        print(f'REFUSED/FAILED: {ex}', file=sys.stderr)
        return 2

    def on_sigint(signum, frame):                  # 1st Ctrl-C: cooperative stop; 2nd: hard interrupt
        if runner.stop_event.is_set():
            raise KeyboardInterrupt
        runner.stop('Ctrl-C')
    signal.signal(signal.SIGINT, on_sigint)
    try:
        summary = runner.run()
    except RunnerError as ex:
        print(f'REFUSED/FAILED: {ex}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    print(json.dumps({k: v for k, v in summary.items() if k not in ('homing',)}, indent=2, default=str))
    return 0 if 'stopped' not in summary else 130


if __name__ == '__main__':
    sys.exit(main())
