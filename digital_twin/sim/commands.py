"""Compile planned joint trajectories into hardware button-pulse schedules and simulate them.

The real excavator only accepts timed button pulses through the GPIO bridge
(hardware/rc-control/src/main.cpp): ``{"cmd": "trigger", "pin": <gpio>, "ms": N}``, one press at a time, with
no joint feedback. This module turns a joint trajectory ``(t, Q)`` into a sequential list of
pulses ``{t_start_s, joint, direction, duration_ms, gpio}`` with
``duration = |dq| / calibrated speed`` (configs/hardware/robot.json), and runs it on

* :class:`SimPulseBackend`  - the MuJoCo twin with env.py's command model (ternary motor
  command, start latency, velocity-limited position target, finite-torque actuator).

The GPIO bridge backend (``BridgePulseBackend``) and the CLI are in digital_twin.hardware.commands,
which re-exports everything here.

All four joints can have pins (hardware_map.json ``joints.<j>.positive`` / ``negative``: ``{pin, label}``,
digital_twin.sim.pins; a joint is available unless its ``status`` says otherwise). Plans are twin trajectories
``[slew, boom, stick, bucket]`` (or hardware-only ``[boom, bucket]`` columns). By default the
compiler emits only boom and bucket (``kinematics.HARDWARE_JOINTS``, what the π0.5 control models
act on) and holds slew / stick at their start value with a warning when the plan moves them;
``compile_schedule(..., joints=JOINTS)`` emits every available joint (the 4-joint policy runner).
Track drive pins are never mapped to a joint.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from ..paths import hardware_map
from . import pins
from .kinematics import FRAMESKIP, HARDWARE_JOINTS, JOINTS, NQ, Arm, full_chain_q, load_config

DRIVE_WORDS = ('drive', 'track')                               # a pin labelled like this is never a joint pin


# ------------------------------------------------------------ hardware map
def load_hardware_map(path=None) -> dict:
    hw = pins.read(path or hardware_map())
    validate_hardware_map(hw)
    return hw


def validate_hardware_map(hw: dict) -> None:
    """Strict: boom and bucket present; only twin joints; every joint pin is ``{pin: int, label: non-empty}``
    (digital_twin.sim.pins), never a drive / track pin; an available joint (status absent or available) has both; no
    GPIO on two joint pins."""
    for name in HARDWARE_JOINTS:
        if hw['joints'].get(name) is None:
            raise ValueError(f'hardware map: missing joint {name}')
    for name, j in hw['joints'].items():
        if name not in JOINTS:
            raise ValueError(f'hardware map: unknown joint {name!r} (twin joints: {", ".join(JOINTS)})')
        status = j.get('status', 'available')
        if status not in ('available', 'locked', 'unavailable'):
            raise ValueError(f'hardware map: {name}.status must be available/locked/unavailable')
        for d in pins.DIRECTIONS:
            p = j.get(d)
            if p is None:
                if status == 'available':
                    raise ValueError(f'hardware map: {name}.{d} missing for an available joint')
                continue
            if not isinstance(p, dict) or isinstance(p.get('pin'), bool) or not isinstance(p.get('pin'), int):
                raise ValueError(f'hardware map: {name}.{d} must be {{"pin": <gpio int>, "label": <name>}}')
            lbl = p.get('label')
            if not isinstance(lbl, str) or not lbl.strip():
                raise ValueError(f'hardware map: {name}.{d} GPIO {p["pin"]} has no label')
            if any(w in lbl.lower() for w in DRIVE_WORDS):
                raise ValueError(f'hardware map: {name}.{d} GPIO {p["pin"]} is a drive pin ({lbl!r})')
    owner = {}
    for name, d, p in pins.joint_drives(hw):
        if 'pin' not in p:
            continue
        gpio = int(p['pin'])
        if gpio in owner:
            raise ValueError(f'hardware map: GPIO {gpio} is mapped to both {owner[gpio]} and {name}.{d}')
        owner[gpio] = f'{name}.{d}'


def emit_joints(hw: dict, joints=HARDWARE_JOINTS) -> list[str]:
    """Joints of ``joints`` (default boom, bucket) the hardware pulses: those the map marks available."""
    return [n for n in joints if (hw['joints'].get(n) or {}).get('status', 'available') == 'available']


def held_joints(hw: dict, joints=HARDWARE_JOINTS) -> list[str]:
    """Twin joints a schedule over ``joints`` holds: every other twin joint and any unavailable one."""
    emit = emit_joints(hw, joints)
    return [n for n in JOINTS if n not in emit]


# ----------------------------------------------------------------- compile
def compile_schedule(t, Q, arm: Arm | None = None, *, windows=None, chunk_s: float | None = None,
                     min_pulse_ms: int | None = None, gap_ms: int | None = None, hw_map: dict | None = None,
                     move_tol_deg: float = 0.5, v_pos=None, v_neg=None, joints=None) -> dict:
    """Compile a joint trajectory into a sequential pulse schedule.

    The trajectory is cut at ``windows`` (end times, e.g. plan segment ends) or every
    ``chunk_s`` seconds; in each window every available joint gets at most one pulse towards
    the window-end configuration. Deltas shorter than ``min_pulse_ms`` are carried into the
    next window (merged). The compiler tracks the commanded state (duration rounded to 1 ms),
    so rounding and merges do not accumulate drift. Pulses never overlap: each starts at
    least ``gap_ms`` after the previous one ends, and not before its window starts.
    ``Q`` columns are the twin's joints ``[slew, boom, stick, bucket]`` or the hardware joints
    ``[boom, bucket]``; ``v_pos`` / ``v_neg`` follow the same columns. ``joints``: the joints that may be
    pulsed (default ``HARDWARE_JOINTS``; ``JOINTS`` = all four, each only when available in the map).
    """
    hw = hw_map or load_hardware_map()
    validate_hardware_map(hw)
    d = hw.get('defaults', {})
    min_pulse_ms = int(d.get('min_pulse_ms', 50) if min_pulse_ms is None else min_pulse_ms)
    gap_ms = int(d.get('gap_ms', 150) if gap_ms is None else gap_ms)
    max_ms = int(d.get('max_pulse_ms', 60000))
    t = np.asarray(t, float); Q = np.asarray(Q, float)
    cols = HARDWARE_JOINTS if Q.shape[1] == len(HARDWARE_JOINTS) else JOINTS
    if v_pos is None or v_neg is None:
        arm = arm or Arm()
        idx = [JOINTS.index(n) for n in cols]
        v_pos, v_neg = arm.v_pos[idx], arm.v_neg[idx]
    emit = [n for n in emit_joints(hw, joints or HARDWARE_JOINTS) if n in cols]
    held = [n for n in cols if n not in emit]
    warnings = []
    for n in held:
        j = cols.index(n)
        need = float(np.degrees(np.max(np.abs(Q[:, j] - Q[0, j]))))
        if need > move_tol_deg:
            why = hw['joints'][n].get('status', 'available') if n in hw['joints'] else 'not pulsed'
            warnings.append(f'{n} is {why} on hardware but the plan moves it by up to '
                            f'{need:.1f} deg; holding it at {np.degrees(Q[0, j]):.1f} deg')
    if windows is None:
        step = chunk_s or (t[-1] - t[0])
        windows = list(np.arange(t[0] + step, t[-1], step))
    ends = sorted({float(w) for w in windows if t[0] < w < t[-1]} | {float(t[-1])})
    starts = [float(t[0])] + ends[:-1]
    q_cmd = Q[0].copy()
    pulses, cursor = [], float(t[0])
    for w0, w1 in zip(starts, ends):
        goal = np.array([np.interp(w1, t, Q[:, j]) for j in range(len(cols))])
        for n in emit:
            j = cols.index(n)
            dq = goal[j] - q_cmd[j]
            direction = 1 if dq >= 0 else -1
            speed = v_pos[j] if direction > 0 else v_neg[j]
            total_ms = int(round(abs(dq) / speed * 1000))
            if total_ms < min_pulse_ms:
                continue                                  # merge into a later window
            jm = hw['joints'][n]
            gpio = pins.entry_pin(jm, direction)
            while total_ms > 0:
                ms = min(total_ms, max_ms)
                start = max(cursor + (gap_ms / 1000 if pulses else 0.0), w0)
                pulses.append({'t_start_s': round(start, 4), 'joint': n, 'direction': direction,
                               'duration_ms': ms, 'gpio': gpio,
                               'label': jm[pins.side(direction)]['label'],
                               'dq_rad': direction * ms / 1000 * speed,
                               'sign_verified': bool(jm.get('sign_verified', True))})
                cursor = start + ms / 1000
                q_cmd[j] += direction * ms / 1000 * speed
                total_ms -= ms
    residual = Q[-1] - q_cmd
    unverified = sorted({p['joint'] for p in pulses if not p['sign_verified']})
    if unverified:
        warnings.append(f'pin direction for {unverified} not verified on hardware (hardware_map sign_verified=false)')
    return {'pulses': pulses, 'warnings': warnings, 'held_joints': held, 'emit_joints': emit, 'joints': list(cols),
            'q_start': Q[0].tolist(), 'q_final_planned': Q[-1].tolist(), 'q_final_commanded': q_cmd.tolist(),
            'residual_deg': dict(zip(cols, np.degrees(residual).tolist())),
            'duration_s': float(cursor - t[0]) if pulses else 0.0, 'plan_duration_s': float(t[-1] - t[0]),
            'min_pulse_ms': min_pulse_ms, 'gap_ms': gap_ms,
            'speeds_rad_s': {'positive': dict(zip(cols, map(float, v_pos))),
                             'negative': dict(zip(cols, map(float, v_neg)))}}


def write_schedule(schedule: dict, output) -> None:
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    (out / 'pulses.json').write_text(json.dumps(schedule, indent=2))
    cols = ['t_start_s', 'joint', 'direction', 'duration_ms', 'gpio', 'label', 'dq_rad', 'sign_verified']
    with open(out / 'pulses.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for p in schedule['pulses']:
            w.writerow({k: p[k] for k in cols})


def _check_pins(schedule: dict) -> None:
    unknown = {p['joint'] for p in schedule['pulses']} - set(HARDWARE_JOINTS)
    if unknown:
        raise PermissionError(f'schedule pulses joints without a pulse channel {sorted(unknown)}; refusing to execute')


# ------------------------------------------------------------ coupled channel model
COUPLED_JOINTS = JOINTS                                        # [slew, boom, stick, bucket]
CHANNELS = ('boom+', 'boom-', 'bucket+', 'bucket-', 'A', 'B')                       # old wiring: A / B yaw AND move the stick
DECOUPLED_CHANNELS = ('boom+', 'boom-', 'bucket+', 'bucket-', 'stick+', 'stick-', 'slew+', 'slew-')   # repaired machine
DUTY_COLUMNS = {'coupled': ('duty_boom', 'duty_bucket', 'duty_A', 'duty_B'),
                'decoupled': ('duty_boom', 'duty_bucket', 'duty_stick', 'duty_slew')}
# Placeholders until hardware_map.json ``coupling`` exists: A = the stick's positive pin, B = its negative pin.
PLACEHOLDER_COUPLING = {'positive': {'yaw_deg_s': 63.0, 'stick_deg_s': 25.0}, 'negative': {'yaw_deg_s': -63.0, 'stick_deg_s': -25.0}}


class ChannelModel:
    """What each button does to [slew, boom, stick, bucket] (rad/s while pressed), one channel at a time.

    boom+/- and bucket+/- move their joint at the calibrated speed; A (GPIO 27) and B (GPIO 22) move
    the slew AND the stick simultaneously at the measured coupling rates (hardware_map.json
    ``coupling.channels``). Every joint but the slew stops at its range limit, so once the stick is at
    its stop the channel keeps yawing: pure yaw. Kinematic (no onset latency / inertia), 1 ms-exact.
    """

    def __init__(self, rates: dict, lo, hi, gpio: dict, onset_s: dict | None = None, source: str = ''):
        self.rates = {k: np.asarray(v, float) for k, v in rates.items()}
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.gpio, self.onset_s, self.source = dict(gpio), dict(onset_s or {}), source
        self.kind = 'decoupled' if 'slew+' in self.rates else 'coupled'
        self.columns = DUTY_COLUMNS[self.kind]

    @classmethod
    def from_config(cls, hw: dict | None = None, joint_cfg: dict | None = None, stick_range=None, warn=print):
        """From hardware_map.json (``coupling`` + calibration via joint_config.real_joint_config) and
        the full4_calibrated joint ranges (``joint_cfg`` {joint: {lo, hi}}; default configs/hardware/robot_full4_calibrated.json)."""
        hw = hw or load_hardware_map()
        from .joint_config import real_joint_config
        real, _ = real_joint_config(hw, None, warn=lambda m: None)
        if joint_cfg is None:
            cfg = load_config('full4_calibrated')['joints']
            joint_cfg = {n: {'lo': -np.inf, 'hi': np.inf} if cfg[n].get('range_deg') is None else
                         dict(zip(('lo', 'hi'), np.radians(cfg[n]['range_deg']))) for n in COUPLED_JOINTS}
        j = {n: i for i, n in enumerate(COUPLED_JOINTS)}
        lo = [joint_cfg[n]['lo'] for n in COUPLED_JOINTS]; hi = [joint_cfg[n]['hi'] for n in COUPLED_JOINTS]
        jm = hw['joints']; st = jm.get('stick') or {}
        sl = hw.get('slew') or {}
        if sl.get('sign_verified', True) and (sl.get('yaw_deg_s') or {}).get('left') and (sl.get('yaw_deg_s') or {}).get('right'):
            # Repaired machine: the stick and the base have their own pins (hardware.calibrate base turn).
            rates, gpio = {}, {}
            for name, joint, sign in (('boom+', 'boom', 1), ('boom-', 'boom', -1), ('bucket+', 'bucket', 1),
                                      ('bucket-', 'bucket', -1), ('stick+', 'stick', 1), ('stick-', 'stick', -1)):
                r = np.zeros(4); r[j[joint]] = sign * (real[joint]['v_pos'] if sign > 0 else real[joint]['v_neg']); rates[name] = r
            for name, key in (('slew+', 'left'), ('slew-', 'right')):
                r = np.zeros(4); r[j['slew']] = np.radians(float(sl['yaw_deg_s'][key])); rates[name] = r
            for name, entry in (('boom', jm['boom']), ('bucket', jm['bucket']), ('stick', st), ('slew', jm['slew'])):
                gpio[f'{name}+'] = pins.entry_pin(entry, 1); gpio[f'{name}-'] = pins.entry_pin(entry, -1)
            onset = {c: float((sl.get('onset_s') or {}).get('left' if c == 'slew+' else 'right') or 0.0)
                     for c in ('slew+', 'slew-')}
            return cls(rates, lo, hi, gpio, onset, 'hardware_map.json slew + calibration (decoupled stick / base)')
        ch = (hw.get('coupling') or {}).get('channels')
        if ch and not all(isinstance((ch.get(d) or {}).get(k), (int, float))
                          for d in pins.DIRECTIONS for k in ('yaw_deg_s', 'stick_deg_s')):
            ch = None                                         # no yaw measured with the stick (the new pins): no A/B model
        source = 'hardware_map.json coupling' if ch else 'placeholders (no coupling section yet)'
        if not ch:
            warn('channel model: hardware_map.json has no coupling section; using placeholder rates')
            ch = PLACEHOLDER_COUPLING
        rates = {}
        for name, joint, sign in (('boom+', 'boom', 1), ('boom-', 'boom', -1), ('bucket+', 'bucket', 1), ('bucket-', 'bucket', -1)):
            r = np.zeros(4); r[j[joint]] = sign * (real[joint]['v_pos'] if sign > 0 else real[joint]['v_neg']); rates[name] = r
        for name, d in (('A', 'positive'), ('B', 'negative')):
            r = np.zeros(4); c = ch[d]
            r[j['slew']] = np.radians(c['yaw_deg_s']); r[j['stick']] = np.radians(c['stick_deg_s']); rates[name] = r
        gpio = {'boom+': pins.entry_pin(jm['boom'], 1), 'boom-': pins.entry_pin(jm['boom'], -1),
                'bucket+': pins.entry_pin(jm['bucket'], 1), 'bucket-': pins.entry_pin(jm['bucket'], -1),
                'A': pins.entry_pin(st, 1) if st else 27, 'B': pins.entry_pin(st, -1) if st else 22}   # 27 / 22: old wiring
        onset = {'A': ch['positive'].get('yaw_onset_s') or 0.0, 'B': ch['negative'].get('yaw_onset_s') or 0.0}
        return cls(rates, lo, hi, gpio, onset, source)

    def step(self, q, channel: str | None, duration_s: float) -> np.ndarray:
        """q after pressing ``channel`` for ``duration_s`` (None / 'none' = nothing pressed)."""
        q = np.asarray(q, float)
        if not channel or channel == 'none' or duration_s <= 0:
            return q.copy()
        return np.clip(q + self.rates[channel] * duration_s, self.lo, self.hi)

    def duty_row(self, channel: str | None, duty: float) -> dict:
        """Signed per-tick duties: duty_boom/duty_bucket (and duty_stick/duty_slew on the repaired machine) in
        [-1, 1]; duty_A/duty_B in [0, 1] on the old wiring."""
        row = {c: 0.0 for c in self.columns}
        if channel and channel != 'none':
            if channel in ('A', 'B'):
                row[f'duty_{channel}'] = duty
            else:
                row[f'duty_{channel[:-1]}'] = duty if channel.endswith('+') else -duty
        return row

    @staticmethod
    def channel_from_duties(duties, min_duty: float = 0.3, kind: str = 'coupled'):
        """[duty_boom, duty_bucket, duty_A, duty_B] (``kind='coupled'``) or [duty_boom, duty_bucket,
        duty_stick, duty_slew] (``'decoupled'``) -> (channel, |duty|) of the largest |duty|, or (None, 0)
        when it is below ``min_duty`` (30 ms of a 100 ms tick)."""
        lo = [-1, -1, 0, 0] if kind == 'coupled' else [-1, -1, -1, -1]
        d = np.clip(np.nan_to_num(np.asarray(duties, float)), lo, [1, 1, 1, 1])
        k = int(np.argmax(np.abs(d)))
        v = float(abs(d[k]))
        if v < min_duty:
            return None, 0.0
        sgn = lambda n: f"{n}{'+' if d[k] > 0 else '-'}"      # noqa: E731
        names = ('boom', 'bucket', 'A', 'B') if kind == 'coupled' else ('boom', 'bucket', 'stick', 'slew')
        return (names[k] if names[k] in ('A', 'B') else sgn(names[k])), v


# ----------------------------------------------------------------- sim backend
class SimPulseBackend:
    """Execute a pulse schedule in the MuJoCo twin with env.py's command model.

    While a pulse is active the joint's position target moves at the calibrated speed
    (``speed_positive/negative_rad_s``, x ``velocity_scale``), clipped 0.025 rad inside the joint
    limits as in env.py; the finite-torque position actuator then drives the joint. Each pulse
    starts ``latency_s`` after it is sent (env.py's command delay; ``randomize`` draws
    100-300 ms like the bridge's link latency and 0.8-1.2 speed scales). Pulses are sent
    sequentially: not before their scheduled time and not before the previous pulse + gap ended.
    Integration is at the 5 ms physics step with fractional overlap, recorded at 20 Hz. Slew and
    stick have no pulses and hold their start values (a hardware-only ``q_start`` [boom, bucket]
    starts them at rest).
    """

    def __init__(self, arm: Arm | None = None, latency_s: float = 0.2, randomize: bool = False, seed: int = 0,
                 settle_s: float = 1.0, velocity_scale=None):
        self.arm = arm or Arm()
        self.latency_s, self.randomize, self.settle_s = latency_s, randomize, settle_s
        self.rng = np.random.default_rng(seed)
        self.velocity_scale = np.ones(NQ) if velocity_scale is None else np.asarray(velocity_scale, float)

    def execute(self, schedule: dict) -> dict:
        _check_pins(schedule)
        arm = self.arm; m, d = arm.model, arm.data
        scale = self.rng.uniform(.8, 1.2, NQ) if self.randomize else self.velocity_scale
        # actual pulse windows
        windows, prev_end = [], -np.inf
        gap = schedule.get('gap_ms', 0) / 1000
        for p in schedule['pulses']:
            send = max(p['t_start_s'], prev_end + gap)
            lat = self.rng.uniform(.1, .3) if self.randomize else self.latency_s
            start = send + lat
            end = start + p['duration_ms'] / 1000
            windows.append((start, end, JOINTS.index(p['joint']), p['direction']))
            prev_end = end
        T = (windows[-1][1] if windows else 0.0) + self.settle_s
        q0 = np.asarray(schedule['q_start'], float)
        if len(q0) == len(HARDWARE_JOINTS):
            q0 = full_chain_q(q0)
        mujoco.mj_resetData(m, d)
        d.qpos[arm.qadr] = q0; d.ctrl[arm.act] = q0
        mujoco.mj_forward(m, d)
        target = q0.copy()
        # 0.025 rad inside the limits as env.py, but never pull a joint off its start value (a
        # held stick sits at its top stop)
        lo, hi = np.minimum(arm.lo + .025, q0), np.maximum(arm.hi - .025, q0)
        rate_pos, rate_neg = arm.v_pos * scale, arm.v_neg * scale
        h = m.opt.timestep
        n_steps = int(np.ceil(T / h))
        rec = {k: [] for k in ('t', 'command', 'target_q', 'qpos', 'tip', 'rot')}
        for s in range(n_steps + 1):
            if s % FRAMESKIP == 0:
                rec['t'].append(s * h); rec['target_q'].append(target.copy())
                rec['qpos'].append(d.qpos[arm.qadr].copy()); rec['tip'].append(arm.point_pos())
                rec['rot'].append(arm.bucket_rot())
                cmd = np.zeros(NQ)
                for (a, b, j, dr) in windows:
                    if a <= s * h < b:
                        cmd[j] = dr
                rec['command'].append(cmd)
            if s == n_steps:
                break
            t0, t1 = s * h, (s + 1) * h
            for (a, b, j, dr) in windows:
                ov = min(b, t1) - max(a, t0)
                if ov > 0:
                    target[j] += dr * (rate_pos[j] if dr > 0 else rate_neg[j]) * ov
            target = np.clip(target, lo, hi)
            d.ctrl[arm.act] = target
            mujoco.mj_step(m, d)
        out = {k: np.array(v) for k, v in rec.items()}
        out.update({'backend': 'sim', 'dry_run': False, 'sent': len(windows),
                    'pulse_windows': [{'start_s': a, 'end_s': b, 'joint': JOINTS[j], 'direction': dr}
                                      for a, b, j, dr in windows],
                    'velocity_scale': scale.tolist(), 'latency_s': self.latency_s, 'randomize': self.randomize})
        return out
