"""Merging a calibration into configs/hardware_map.json: re-read right before each write, validated, written
atomically (then digital_twin.sim.kinematics' cached reads are dropped)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from digital_twin.sim import pins

from .rig import joint_pins


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def merge_calibration(hw: dict, cal: dict, session: str, signs_only: bool = False) -> dict:
    """A run's results into a hardware map dict: the pin signs (``joints.<j>.sign_verified``: absent = verified,
    false = reversed; read by hardware.bridge), and unless ``signs_only``: ``calibration.arm_imus`` (the
    estimator's calibration), the measured ranges and speeds with their press-to-motion ``dead_time_s``
    (``calibration.ranges`` / ``speeds``; digital_twin.sim.joint_config, the runner's onsets) and the base turn (the
    top-level ``slew``: yaw_deg_s, onset_s, sign; ``calibration.slew``). Other entries are kept; an older map's pin
    keys (and an older run's checks keyed by GPIO) come out in the current shape (digital_twin.sim.pins.migrate)."""
    pins.migrate(hw)
    src = f'digital_twin.hardware.calibrate (link IMUs, {session})'
    for j, s in (cal.get('signs') or {}).items():
        jm = joint_pins(hw, j)
        if s.get('sign') == -1:
            jm['sign_verified'] = False                        # the map records only failed checks
        elif s.get('verified'):
            jm.pop('sign_verified', None); jm['sign'] = 1
    if signs_only:
        return hw
    c = hw.setdefault('calibration', {})
    c['arm_imus'] = {'session': session, **{k: v for k, v in cal.items() if k not in ('failures', 'slew')}}
    rng = c.setdefault('ranges', {})
    for j, r in cal['ranges_rad'].items():
        rng[j] = {**(rng.get(j) or {}), 'range_measured_rad': float(r), 'direction_ok': True,
                  'range_spread_deg': cal['checks']['joints'][j].get('range_spread_deg'), 'source': src}
    # speeds straight from each link's own gyro (end-to-end drives), so no slack of the joints below it
    spd = c.setdefault('speeds', {})
    for j, v in (cal.get('speeds_rad_s') or {}).items():
        v = {k: float(x) for k, x in v.items() if x}
        if v:
            spd[j] = {**(spd.get(j) or {}), 'speed_measured_rad_s': v, 'confidence': {k: 'medium' for k in v},
                      'untrusted_estimates_rad_s': {}, 'source': src}
    for j, d in (cal.get('dead_time_s') or {}).items():
        e = spd.setdefault(j, {})
        e['dead_time_s'] = {**(e.get('dead_time_s') or {}), **{k: float(x) for k, x in d.items() if x is not None}}
    s = cal.get('slew') or {}
    if s.get('slew'):
        sl = hw.setdefault('slew', {})
        if s['sign']:
            sl.update({'sign': s['sign'],
                       'yaw_deg_s': {w: s['slew'][w]['yaw_rate_deg_s'] for w in ('left', 'right')},
                       'mean_yaw_deg_s': {w: s['slew'][w]['mean_yaw_deg_s'] for w in ('left', 'right')},
                       'onset_s': {w: s['slew'][w]['onset_s'] for w in ('left', 'right')},
                       'reference': 'slew 0 = base facing forward at the start of the base turn', 'source': src})
        if s['sign_verified']:
            sl.pop('sign_verified', None)
        else:                                                  # no clear turn keeps the old rates, flagged
            sl['sign_verified'] = False
        c['slew'] = {'net_yaw_deg': s['net_yaw_deg'], 'reason': s['reason'], 'session': session}
    return pins.migrate(hw)                                   # an older run's checks keyed by GPIO


def update_hardware_map(path: Path, mutate) -> dict:
    """Re-read the map right before writing, apply ``mutate(hw)``, validate, write atomically; then drop the
    cached reads of the map (kinematics.reload) so locked_joints() / rest_q() see the new values."""
    from digital_twin.sim import kinematics
    from digital_twin.sim.commands import validate_hardware_map
    path = Path(path)
    hw = pins.read(path)
    mutate(hw)
    validate_hardware_map(hw)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(_jsonable(hw), indent=2) + '\n')
    tmp.replace(path)
    kinematics.reload()
    return hw


def write_hardware_map(path: Path, cal: dict, session: str, signs_only: bool = False) -> dict:
    return update_hardware_map(path, lambda hw: merge_calibration(hw, cal, session, signs_only))
