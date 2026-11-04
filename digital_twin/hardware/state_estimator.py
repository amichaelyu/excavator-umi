"""Joint-state estimators for the closed-loop policy runner (digital_twin.policy.runner).

The real excavator has no joint encoders: it only takes timed button pulses. Without the arm IMUs the
estimate is the integral of the pulses we command, at the calibrated joint speeds.

* :class:`DeadReckoningEstimator` - integrates commanded pulses at the calibrated per-joint,
  per-direction speeds (configs/sim/robot.json ``speed_positive/negative_rad_s``, falling back to
  ``speed_rad_s``) and clips to the joint ranges (``range_deg``, the same numbers build.py writes
  into the MuJoCo models). Pure numpy: no MuJoCo, usable on the robot laptop.
* :class:`TwinEstimator` - reads the MuJoCo twin's qpos from the runner's sim executor (sim runs).
* :class:`ArmIMUEstimator` - the IMU estimate: one IMU per link (boom, stick, bucket), each link's absolute
  pitch from the rotation of its gravity reading about its calibrated joint axis (gyro complementary filter),
  joints = differences of neighbouring link pitches. Measures boom, stick and bucket at inference
  time and writes them into the base estimator. Calibrated by digital_twin.hardware.calibrate
  (configs/hardware_map.json ``calibration.arm_imus``).
* :class:`SlewMock` - the slew is locked and not estimated: the 4-joint policy is fed back the slew it last
  asked for, and its slew targets are dropped so the base never turns.

External estimators (AprilTag, VIO from the bucket camera, ...) plug in by subclassing
:class:`StateEstimator` (``reset`` / ``update_from_pulses`` / ``joint_state``) or, for a measured
absolute joint value, by calling :meth:`DeadReckoningEstimator.set_joint` from the runner's
``external_hook`` (see policy.runner.PolicyRunner.external_hook).

Joint vector: the runner's ``cloak_full4`` profile tracks :data:`FULL4_JOINTS` (``joints=`` of
:class:`DeadReckoningEstimator`) and reads its state with ``policy_state(t, FULL4_JOINTS)``. The default,
:data:`POLICY_JOINTS` = ``('boom', 'bucket')`` (= kinematics.HARDWARE_JOINTS), plus ``stick`` is enough to
dead-reckon homing.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from digital_twin.sim.joint_config import POLICY_JOINTS, TRACKED_JOINTS, load_joint_config

FULL4_JOINTS = ('slew', 'boom', 'stick', 'bucket')    # = kinematics.JOINTS, the 4-joint profile's state / action order


class StateEstimator(ABC):
    """Interface: ``reset(q0)``, ``update_from_pulses(pulses, t)``, ``joint_state(t)``.

    ``q0`` is a dict {joint: rad} (or a [boom, bucket] array). ``pulses`` are the compiled pulse
    dicts of digital_twin.hardware.commands (``t_start_s`` relative to ``t``, ``joint``, ``direction``,
    ``duration_ms``); ``t`` is the absolute time [s] (sim time or time.monotonic()) of schedule
    time 0. ``joint_state(t)`` returns the estimate at ``t`` in :attr:`joints` order.
    """
    joints: tuple[str, ...] = TRACKED_JOINTS

    @abstractmethod
    def reset(self, q0) -> None: ...

    @abstractmethod
    def update_from_pulses(self, pulses, t: float) -> None: ...

    @abstractmethod
    def joint_state(self, t: float) -> np.ndarray: ...

    def get(self, name: str, t: float) -> float:
        return float(self.joint_state(t)[self.joints.index(name)])

    def policy_state(self, t: float, joints=POLICY_JOINTS) -> np.ndarray:
        """``joints`` in that order; default [boom, bucket]."""
        q = self.joint_state(t)
        return np.array([q[self.joints.index(n)] for n in joints])

    def as_dict(self, t: float) -> dict[str, float]:
        return dict(zip(self.joints, map(float, self.joint_state(t))))


def _as_dict(q0, joints) -> dict[str, float]:
    if isinstance(q0, dict):
        return {k: float(v) for k, v in q0.items()}
    q0 = np.asarray(q0, float).ravel()
    return dict(zip(POLICY_JOINTS if len(q0) == len(POLICY_JOINTS) else joints, map(float, q0)))


class DeadReckoningEstimator(StateEstimator):
    """Integrates commanded pulses at the calibrated per-joint, per-direction speeds.

    Each pulse moves its joint at the calibrated speed for its duration (optionally starting
    ``latency_s`` late), clipped to the joint range after every pulse, so driving into a hard stop
    saturates at the limit exactly like the MuJoCo joint limit / the real mechanical stop.
    Pulses are sequential (one pin at a time), so integrating them in start order is exact.
    ``joints`` (default :data:`TRACKED_JOINTS`) are the joints tracked; ``joint_cfg`` must cover them.
    """

    def __init__(self, q0=None, joint_cfg: dict | None = None, latency_s: float = 0.0, joints=None):
        if joints is not None:
            self.joints = tuple(joints)
        self.cfg = joint_cfg or load_joint_config(joints=self.joints)
        self.latency_s = float(latency_s)
        self.reset(q0 if q0 is not None else {})

    def reset(self, q0) -> None:
        d = _as_dict(q0, self.joints)
        self._q = np.array([self._clip(n, d.get(n, 0.0)) for n in self.joints])
        self._segments: list[tuple[float, float, int, float]] = []   # (t0, t1, joint index, rate)

    def _clip(self, name, v):
        c = self.cfg[name]
        return float(min(max(v, c['lo']), c['hi']))

    def set_joint(self, name: str, value: float, t: float | None = None) -> None:
        """Overwrite one joint (external measurement / IMU correction). Pending pulses of that
        joint that already started are committed first; later ones still apply."""
        j = self.joints.index(name)
        if t is not None:
            self.joint_state(t)
            # a pulse of this joint that is running at t continues only from t on
            self._segments = [(max(t0, t), t1, jj, r) if jj == j and t0 < t else (t0, t1, jj, r)
                              for t0, t1, jj, r in self._segments]
        self._q[j] = self._clip(name, value)

    def update_from_pulses(self, pulses, t: float) -> None:
        for p in pulses:
            if p['joint'] not in self.joints:
                raise ValueError(f"estimator does not track joint {p['joint']!r}")
            j = self.joints.index(p['joint'])
            c = self.cfg[p['joint']]
            rate = p['direction'] * (c['v_pos'] if p['direction'] > 0 else c['v_neg'])
            t0 = t + float(p['t_start_s']) + self.latency_s
            self._segments.append((t0, t0 + p['duration_ms'] / 1000.0, j, rate))
        self._segments.sort(key=lambda s: s[0])

    def joint_state(self, t: float) -> np.ndarray:
        # commit finished segments (in order), then evaluate running ones on a copy
        while self._segments and self._segments[0][1] <= t:
            t0, t1, j, rate = self._segments.pop(0)
            self._q[j] = self._clip(self.joints[j], self._q[j] + rate * (t1 - t0))
        q = self._q.copy()
        for t0, t1, j, rate in self._segments:
            if t > t0:
                q[j] = self._clip(self.joints[j], q[j] + rate * (min(t, t1) - t0))
        return q

    def pending_until(self) -> float:
        return max((s[1] for s in self._segments), default=-math.inf)


class TwinEstimator(StateEstimator):
    """Reads the MuJoCo twin's joint positions (sim runs). ``sim`` needs ``qpos_by_name()``
    returning {joint: rad}; welded joints come from kinematics.locked_joints()."""

    def __init__(self, sim, welded: dict | None = None):
        self.sim = sim
        self.welded = dict(welded or {})

    def reset(self, q0) -> None:
        pass                                            # the sim itself is reset / homed

    def update_from_pulses(self, pulses, t: float) -> None:
        pass                                            # physics integrates them

    def joint_state(self, t: float) -> np.ndarray:
        q = {**self.welded, **self.sim.qpos_by_name()}
        return np.array([float(q.get(n, 0.0)) for n in self.joints])


# ------------------------------------------------------------------------------ arm IMUs (one per link)
ARM_LINKS = ('boom', 'stick', 'bucket')                  # IMU names = the link they ride on, base -> tip
LINK_PARENT = {'boom': None, 'stick': 'boom', 'bucket': 'stick'}


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError('zero / invalid vector')
    return v / n


def link_pitches(q: dict) -> dict[str, float]:
    """Absolute link pitches of the planar chain (twin convention: bucket pitch = boom + stick + bucket)."""
    out, acc = {}, 0.0
    for n in ARM_LINKS:
        acc += float(q[n])
        out[n] = acc
    return out


def joints_from_pitches(pitch: dict) -> dict[str, float]:
    """Inverse of :func:`link_pitches` for the links present (a joint needs its link and its parent's)."""
    out = {}
    for n in ARM_LINKS:
        par = LINK_PARENT[n]
        if n in pitch and (par is None or par in pitch):
            out[n] = float(pitch[n] - (pitch[par] if par else 0.0))
    return out


def link_rotation(up_ref, up, axis) -> float:
    """Angle [rad] the link turned about ``axis`` (IMU frame, right-handed) between two at-rest readings.

    An accelerometer at rest reads the world up direction in its own frame. When the link turns by
    +theta about its axis, that vector turns by -theta in the sensor frame, so theta is the signed
    angle from ``up`` back to ``up_ref``, both projected onto the plane normal to the axis.
    """
    a = unit(axis)
    p0 = np.asarray(up, float) - np.dot(up, a) * a
    p1 = np.asarray(up_ref, float) - np.dot(up_ref, a) * a
    return float(math.atan2(np.dot(a, np.cross(p0, p1)), np.dot(p0, p1)))


def axis_from_arcs(arcs) -> tuple[np.ndarray, float, float]:
    """Joint axis in one IMU frame from arcs of at-rest up readings, each taken while the link turned
    about that axis: ``[{'points': [up, ...], 'sign': s}]`` with ``s`` the known sign of the link's
    rotation from the first point to the one farthest from it.

    Every reading of a rotation about ``a`` keeps the same component along ``a``, so the points of an
    arc lie in a plane normal to it: ``a`` is the least-variance direction of the per-arc centred
    points (an arc needs 3+ distinct points, or two arcs in different directions). Returns (axis
    oriented so the link turns + about it, out-of-plane residual [rad] = the largest distance of a
    point from its arc's plane, ambiguity = sqrt(smallest / middle eigenvalue): ~0 for a well-defined
    plane, ~1 when the points do not span one, e.g. only two distinct readings).
    """
    S = np.zeros((3, 3))
    clouds = []
    for arc in arcs:
        P = np.array([unit(u) for u in arc['points']])
        C = P - P.mean(0)
        S += C.T @ C
        clouds.append(C)
    w, V = np.linalg.eigh(S)
    a = V[:, 0]
    vote = 0.0
    for arc in arcs:
        P = [unit(u) for u in arc['points']]
        far = max(P, key=lambda u: float(np.linalg.norm(u - P[0])))
        vote += np.sign(arc['sign']) * link_rotation(P[0], far, a)
    if vote < 0:
        a = -a
    resid = max(float(np.max(np.abs(C @ a))) for C in clouds)
    amb = 1.0 if w[1] <= 1e-12 * max(w[2], 1e-18) else math.sqrt(max(w[0], 0.0) / w[1])
    return a, math.asin(min(1.0, resid)), float(amb)


class ArmIMUEstimator(StateEstimator):
    """Joint state from one IMU per link (``ARM_LINKS``), on top of a base estimator (dead reckoning).

    Calibration (``calibration.arm_imus`` of configs/hardware_map.json, from
    digital_twin.hardware.calibrate) gives per IMU the joint axis in its frame (``axis_imu``), its
    at-rest up reading ``up_ref_imu`` at the reference pose ``reference_q_rad`` (every joint at its
    home end) and its gyro bias at rest (``gyro_bias_dps``, subtracted before the gyro predicts). A link's pitch is then ``pitch_ref + link_rotation(up_ref, up, axis) + offset``:
    no mounting rotation is needed, only the axis. ``offset`` is common to all links: a base standing
    on a different slope (tilt about the joint axis) shifts every link pitch equally, so
    :meth:`rereference_boom` re-anchors it once the boom is at a known angle (homed). Stick and bucket
    are differences of link pitches and never depend on it.

    ``update(readings, t)`` feeds ``{link: (accel_g, gyro_dps, t_sample)}``. Each link runs a
    complementary filter: the gyro rate about the axis predicts, the gravity pitch corrects (time
    constant ``tau_s``); a reading whose magnitude is off 1 g by more than ``max_dyn_g`` (the arm is
    accelerating) only propagates the gyro. After a gap longer than ``max_gap_s`` (the policy loop reads
    once per cycle, so one gyro sample says nothing about the motion since) the gravity pitch is taken as is. A joint is measured while its link and its parent link
    are fresh (``max_age_s``); others keep the base value. :meth:`apply_correction` writes the
    measured joints into the base, so the policy state is the measured state and later pulses
    integrate from it.
    """

    def __init__(self, base: StateEstimator, cal: dict, tau_s: float = 0.25, max_age_s: float = 0.5,
                 max_dyn_g: float = 0.2, max_gap_s: float = 0.1, correct: bool = True):
        self.base, self.joints = base, base.joints
        imus = cal['imus']
        missing = [n for n in ARM_LINKS if n not in imus]
        if missing:
            raise ValueError(f'arm IMU calibration lacks {missing}')
        self.axis = {n: unit(imus[n]['axis_imu']) for n in ARM_LINKS}
        self.up_ref = {n: unit(imus[n]['up_ref_imu']) for n in ARM_LINKS}
        bias = cal.get('gyro_bias_dps') or {}
        self.gyro_bias = {n: np.asarray(bias.get(n, (0.0, 0.0, 0.0)), float) for n in ARM_LINKS}
        self.pitch_ref = link_pitches({n: float(cal['reference_q_rad'][n]) for n in ARM_LINKS})
        stops = cal.get('stops_rad') or {}
        self.stops = {n: (float(stops[n][0]), float(stops[n][1])) for n in stops}
        self.tau_s, self.max_age_s, self.max_dyn_g = float(tau_s), float(max_age_s), float(max_dyn_g)
        self.max_gap_s = float(max_gap_s)
        self.correct = bool(correct)
        self.calibrated = True
        self.offset = 0.0
        self.pitch: dict[str, float] = {}
        self.t_imu: dict[str, float] = {}
        self.measured_at: float | None = None

    @classmethod
    def from_hardware_map(cls, base, hw: dict, **kw):
        cal = (hw.get('calibration') or {}).get('arm_imus')
        if not cal:
            raise ValueError('configs/hardware_map.json has no calibration.arm_imus; run digital_twin.hardware.calibrate')
        return cls(base, cal, **kw)

    # -- StateEstimator
    def reset(self, q0) -> None:
        self.base.reset(q0)

    def update_from_pulses(self, pulses, t: float) -> None:
        self.base.update_from_pulses(pulses, t)

    # -- measurements
    def raw_pitch(self, link: str, accel) -> float:
        """Gravity-only link pitch (no filter) from one at-rest accel reading."""
        return self.pitch_ref[link] + link_rotation(self.up_ref[link], unit(accel), self.axis[link]) + self.offset

    def update(self, readings: dict, t: float | None = None) -> dict[str, float]:
        for link, r in readings.items():
            if link not in self.axis or r is None:
                continue
            accel, gyro, ts = r
            ts = t if ts is None else float(ts)
            if self.t_imu.get(link) is not None and ts is not None and ts <= self.t_imu[link]:
                continue                                          # same sample polled twice
            n = float(np.linalg.norm(accel))
            still = abs(n - 1.0) <= self.max_dyn_g
            meas = self.raw_pitch(link, accel) if n > 1e-6 else None
            prev, tp = self.pitch.get(link), self.t_imu.get(link)
            if prev is None or tp is None or ts is None or ts - tp > self.max_gap_s:
                if meas is None or (prev is None and not still):
                    continue
                self.pitch[link] = meas
            else:
                dt = max(0.0, ts - tp)
                rate = 0.0 if gyro is None else math.radians(float(np.dot(np.asarray(gyro, float) - self.gyro_bias[link],
                                                                          self.axis[link])))
                pred = prev + rate * dt
                if meas is not None and still:
                    a = self.tau_s / (self.tau_s + dt) if dt > 0 else 1.0
                    pred += (1 - a) * ((meas - pred + math.pi) % (2 * math.pi) - math.pi)
                self.pitch[link] = pred
            self.t_imu[link] = ts
        return dict(self.pitch)

    def set_static(self, accels: dict, t: float) -> dict[str, float]:
        """Restart the filters from at-rest (averaged) accel readings {link: accel_g} taken at ``t``."""
        for link, a in accels.items():
            if link in self.axis:
                self.pitch[link] = self.raw_pitch(link, a)
                self.t_imu[link] = t
        return dict(self.pitch)

    def fresh_pitches(self, t: float) -> dict[str, float]:
        return {n: p for n, p in self.pitch.items()
                if self.t_imu.get(n) is None or t - self.t_imu[n] <= self.max_age_s}

    def measured(self, t: float) -> dict[str, float]:
        """{joint: rad} for every joint whose link and parent-link IMUs are fresh."""
        q = joints_from_pitches(self.fresh_pitches(t))
        return {n: v for n, v in q.items() if n in self.joints}

    def rereference_boom(self, boom_rad: float, t: float) -> float | None:
        """The boom is known to be at ``boom_rad`` (on its homed stop): fold the difference into the
        common offset (base tilt about the joint axis). Returns the shift [rad] or None without a reading."""
        p = self.fresh_pitches(t).get('boom')
        if p is None:
            return None
        d = float(boom_rad) - p
        self.offset += d
        self.pitch = {n: v + d for n, v in self.pitch.items()}
        return d

    def joint_state(self, t: float) -> np.ndarray:
        q = self.base.joint_state(t)
        if self.correct:
            for n, v in self.measured(t).items():
                q[self.joints.index(n)] = v
        return q

    def apply_correction(self, t: float) -> dict[str, float]:
        """Write the measured joints into the base estimator; returns what was written."""
        if not self.correct or not hasattr(self.base, 'set_joint'):
            return {}
        m = self.measured(t)
        for n, v in m.items():
            self.base.set_joint(n, v, t)
        if m:
            self.measured_at = t
        return m


# ------------------------------------------------------------------------------ coupled machine
COUPLED_JOINTS = ('slew', 'boom', 'stick', 'bucket')     # = commands.COUPLED_JOINTS = kinematics.JOINTS


class CoupledDeadReckoningEstimator(StateEstimator):
    """Dead reckoning over [slew, boom, stick, bucket] from *channel* pulses (commands.ChannelModel).

    A pulse is ``{'t_start_s', 'channel', 'duration_ms'}`` with channel in boom+/-, bucket+/-, A, B;
    A/B move the slew and the stick together, the stick saturates at its stop while the slew keeps
    turning. ``rates`` / ``lo`` / ``hi`` come from ChannelModel (``from_channel_model``) so this module
    stays MuJoCo-free. ``set_joint`` accepts external measurements (e.g. slew from :class:`YawEstimator`).
    """
    joints = COUPLED_JOINTS

    def __init__(self, rates: dict, lo, hi, q0=None):
        self.rates = {k: np.asarray(v, float) for k, v in rates.items()}
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.reset(q0 if q0 is not None else {})

    @classmethod
    def from_channel_model(cls, cm, q0=None):
        return cls(cm.rates, cm.lo, cm.hi, q0)

    def reset(self, q0) -> None:
        d = q0 if isinstance(q0, dict) else dict(zip(self.joints, np.asarray(q0, float).ravel()))
        self._q = np.clip(np.array([float(d.get(n, 0.0)) for n in self.joints]), self.lo, self.hi)
        self._segments = []

    def set_joint(self, name: str, value: float, t: float | None = None) -> None:
        if t is not None:
            self.joint_state(t)
        j = self.joints.index(name)
        self._q[j] = float(np.clip(value, self.lo[j], self.hi[j]))

    def update_from_pulses(self, pulses, t: float) -> None:
        for p in pulses:
            ch = p.get('channel') or (f"{p['joint']}{'+' if p['direction'] > 0 else '-'}" if p.get('joint') in ('boom', 'bucket')
                                      else p.get('joint'))
            if ch not in self.rates:
                raise ValueError(f'unknown channel {ch!r}')
            t0 = t + float(p['t_start_s'])
            self._segments.append((t0, t0 + p['duration_ms'] / 1000.0, ch))
        self._segments.sort(key=lambda s: s[0])

    def joint_state(self, t: float) -> np.ndarray:
        while self._segments and self._segments[0][1] <= t:
            t0, t1, ch = self._segments.pop(0)
            self._q = np.clip(self._q + self.rates[ch] * (t1 - t0), self.lo, self.hi)
        q = self._q.copy()
        for t0, t1, ch in self._segments:
            if t > t0:
                q = np.clip(q + self.rates[ch] * (min(t, t1) - t0), self.lo, self.hi)
        return q

    def policy_state(self, t: float, joints=None) -> np.ndarray:
        q = self.joint_state(t)                          # the pulse profile's state is all four joints
        return q if joints is None else np.array([q[self.joints.index(n)] for n in joints])


class SlewMock:
    """Locked slew for the 4-joint policy: the base never turns and the slew is not estimated.

    The policy's slew state is mocked as the slew it last asked for (:meth:`state`; ``desired``, 0 = facing
    forward at the start), and its slew targets are replaced by the real, locked slew before the pulses are
    compiled (:meth:`lock`), so no slew pin is driven. :meth:`command` records the newest desired slew, clipped
    to ``lo`` / ``hi``. The real slew (for the robot mask) stays whatever the base estimator holds.
    """

    def __init__(self, joints, desired: float = 0.0, lo: float = -math.inf, hi: float = math.inf):
        self.i = tuple(joints).index('slew')
        self.lo, self.hi = float(lo), float(hi)
        self.reset(desired)

    def reset(self, desired: float = 0.0) -> None:
        self.desired = float(min(max(desired, self.lo), self.hi))

    def state(self, q) -> np.ndarray:
        """``q`` with the slew replaced by the mocked (last desired) slew."""
        q = np.array(q, float)
        q[self.i] = self.desired
        return q

    def lock(self, Q, q_real) -> np.ndarray:
        """Targets ``Q`` (rows) with the slew column held at the real slew of ``q_real``: no slew motion."""
        Q = np.array(Q, float)
        Q[..., self.i] = float(np.asarray(q_real, float)[self.i])
        return Q

    def command(self, Q) -> float:
        """Record the newest desired slew from targets ``Q`` (the last row) and return it."""
        Q = np.atleast_2d(np.asarray(Q, float))
        if len(Q):
            self.reset(Q[-1, self.i])
        return self.desired


class YawEstimator:
    """Base yaw (slew) from the bucket IMU gyro (policy rollout display only; the policy runner locks the slew
    and mocks it with :class:`SlewMock`): the rate about the measured gravity direction
    (``omega . up``, IMU frame, so no mount calibration is needed), bias-corrected while at rest,
    integrated over time. + = counter-clockwise seen from above (hardware_map coupling convention).

    At rest (|gyro| < ``rest_dps`` and |accel| within ``rest_g`` of 1 g for ``rest_s``), the bias is a
    running mean of the up-rate and the integration is frozen. Boom/bucket motion tilts the IMU but
    does not rotate it about the vertical, so the vertical rate is the base yaw rate.
    """

    def __init__(self, yaw0: float = 0.0, rest_dps: float = 1.5, rest_g: float = 0.03, rest_s: float = 0.5,
                 bias_tau_s: float = 5.0):
        self.yaw, self.bias = float(yaw0), 0.0
        self.rest_dps, self.rest_g, self.rest_s, self.bias_tau_s = rest_dps, rest_g, rest_s, bias_tau_s
        self.t = None
        self._rest_since = None
        self.at_rest = False

    def reset(self, yaw0: float = 0.0):
        self.yaw = float(yaw0)

    def update(self, accel_g, gyro_dps, t: float) -> float:
        a = np.asarray(accel_g, float); g = np.asarray(gyro_dps, float)
        n = np.linalg.norm(a)
        up = a / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        rate = math.radians(float(g @ up))
        still = np.linalg.norm(g) < self.rest_dps and abs(n - 1.0) < self.rest_g
        if still:
            self._rest_since = t if self._rest_since is None else self._rest_since
        else:
            self._rest_since = None
        self.at_rest = self._rest_since is not None and t - self._rest_since >= self.rest_s
        if self.t is not None:
            dt = max(0.0, t - self.t)
            if self.at_rest:
                k = min(1.0, dt / self.bias_tau_s) if self.bias_tau_s > 0 else 1.0
                self.bias += k * (rate - self.bias)
            else:
                self.yaw += (rate - self.bias) * dt
        self.t = t
        return self.yaw
