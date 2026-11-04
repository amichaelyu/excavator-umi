"""Per-joint ranges and speeds ``{lo, hi, v_pos, v_neg}`` (rad, rad/s), with no MuJoCo.

Priors come from configs/hardware/robot.json; :func:`real_joint_config` replaces them with the calibration measured
on the machine (the hardware map's ``calibration``, digital_twin.paths.hardware_map()), anchored at the homing stops. The calibrated
twin (``build --calibrated``), the pulse channel model and the policy runner's joint config all use it.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..paths import CONFIG


POLICY_JOINTS = ('boom', 'bucket')                  # must match kinematics.HARDWARE_JOINTS
TRACKED_JOINTS = ('boom', 'bucket', 'stick')


def load_joint_config(robot_json=None, joints=TRACKED_JOINTS) -> dict[str, dict]:
    """{joint: {lo, hi, v_pos, v_neg}} (rad, rad/s) from configs/hardware/robot.json (no MuJoCo needed)."""
    cfg = json.loads(Path(robot_json or CONFIG / 'robot.json').read_text())['joints']
    out = {}
    for n in joints:
        j = cfg[n]
        lo, hi = np.radians(j['range_deg'])
        out[n] = {'lo': float(lo), 'hi': float(hi),
                  'v_pos': float(j.get('speed_positive_rad_s', j['speed_rad_s'])),
                  'v_neg': float(j.get('speed_negative_rad_s', j['speed_rad_s']))}
    return out


def real_joint_config(hw: dict, prior_cfg: dict | None = None, warn=print) -> tuple[dict, dict]:
    """({joint: {lo, hi, v_pos, v_neg}} in real joint space, {joint: source notes}).

    Ranges: measured range anchored at the homing stop; speeds: measured per direction."""
    prior = copy.deepcopy(prior_cfg or load_joint_config())
    cal = hw.get('calibration') or {}
    ranges, speeds = cal.get('ranges') or {}, cal.get('speeds') or {}
    homing = (hw.get('homing') or {}).get('joints') or {}
    out, src = {}, {}
    for n in TRACKED_JOINTS:
        c = dict(prior[n]); notes = []
        h = homing.get(n)
        r = (ranges.get(n) or {}).get('range_measured_rad')
        if h is not None and r is not None and r > 0:
            stop, d = float(h['home_value_rad']), int(h['home_direction'])
            c['lo'], c['hi'] = (stop - r, stop) if d > 0 else (stop, stop + r)
            notes.append('range: measured')
        else:
            notes.append('range: prior')
            if n in POLICY_JOINTS:
                warn(f'joint config: no measured range for {n}; using the prior range')
        sp = (speeds.get(n) or {}).get('speed_measured_rad_s') or {}
        meas = {name: float(sp[name]) for name in ('positive', 'negative') if sp.get(name) is not None and sp[name] > 0}
        for key, name, other in (('v_pos', 'positive', 'negative'), ('v_neg', 'negative', 'positive')):
            if name in meas:
                c[key] = meas[name]
            elif other in meas:
                # one direction measured: the same motor and gearing, so it beats the video prior
                c[key] = meas[other]
                if n in POLICY_JOINTS:
                    warn(f'joint config: no measured {name} speed for {n}; using its measured {other} '
                         f'speed {meas[other]:.3f} rad/s')
            elif n in POLICY_JOINTS:
                warn(f'joint config: no measured {name} speed for {n}; using the prior {c[key]:.3f} rad/s')
        notes.append('speed: ' + ('measured' if len(meas) == 2 else 'measured (one direction)' if meas else 'prior'))
        out[n], src[n] = c, ', '.join(notes)
    return out, src
