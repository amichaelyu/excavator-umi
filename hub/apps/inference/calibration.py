"""Hardware calibration as the inference page shows it, read fresh on every call (no side effects): what
configs/hardware_map.json holds from ``digital_twin.hardware.calibrate`` (the link-IMU calibration, ranges,
speeds and dead times, the base turn, pin signs: :func:`map_status`, :func:`summary_line`) and the newest runs
under artifacts/arm-imu-calibration/<stamp>[-sim]/ (``calibration.json``, ``log.txt``: :func:`last_run`,
:func:`cal_report`). Running it is :class:`hub.apps.inference.calibration_jobs.CalibrationJobs`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from hub.apps.inference.util import load_json
from digital_twin.hardware.calibrate import ARTIFACTS as CAL_ROOT
from digital_twin.paths import REPO
from digital_twin.sim import pins

CAL_JOINTS = ('boom', 'stick', 'bucket')
DIR_NAMES = {'boom': {'positive': 'up', 'negative': 'down'}, 'stick': {'positive': 'up', 'negative': 'down'},
             'bucket': {'positive': 'out', 'negative': 'in'}}
RUN_RE = re.compile(r'^\d{8}T\d{6}Z(-sim)?$')
MANUAL_PREREQS = ['base facing forward (slew 0: the base turn starts there and comes back)',
                  'arm clear of people and objects, and of the ground with the boom all the way down and the stick and '
                  'bucket out', 'the boom, stick and bucket IMUs live on the Pi dashboard']
MOTION = ('Homes every joint, then drives the bucket, stick and boom end to end (2 round trips each; the boom with the '
          'stick and bucket out, so the arm reaches full boom down, arm out), homes again, then turns the base left and '
          'right (300 / 600 ms presses, each undone by the other pin). A few minutes; writes the hardware map when live.')


def _deg(v, nd=1):
    return None if v is None else round(math.degrees(float(v)), nd)


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO) if p.is_relative_to(REPO) else p)


def fmt_stamp(s):
    """'20260927T152434Z' -> '2026-09-27 15:24:34Z' (other strings unchanged)."""
    m = re.match(r'^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)(\d\d)Z', str(s or ''))
    return f'{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}Z' if m else s


def map_status(path) -> dict:
    """What configs/hardware_map.json holds from the calibration (read fresh every call)."""
    path = Path(path)
    hw = pins.read(path)
    cal = hw.get('calibration') or {}
    arm = cal.get('arm_imus') or {}
    joints = {}
    for j in CAL_JOINTS:
        jm = pins.joint_entry(hw, j) or {}
        sp = (cal.get('speeds') or {}).get(j) or {}
        joints[j] = {**{d: jm.get(d) for d in pins.DIRECTIONS},                 # {pin, label} per direction
                     'sign_verified': bool(jm.get('sign_verified', True)),
                     'range_deg': _deg(((cal.get('ranges') or {}).get(j) or {}).get('range_measured_rad')),
                     'speed_deg_s': {DIR_NAMES[j][k]: _deg(v) for k, v in (sp.get('speed_measured_rad_s') or {}).items()
                                     if k in DIR_NAMES[j] and v},
                     'dead_time_s': {DIR_NAMES[j][k]: round(float(v), 3) for k, v in (sp.get('dead_time_s') or {}).items()
                                     if k in DIR_NAMES[j] and isinstance(v, (int, float)) and v > 0}}
    sl = hw.get('slew') or {}
    y = sl.get('yaw_deg_s') or {}
    return {'path': _rel(path), 'mtime': path.stat().st_mtime,
            'arm_imus': {'calibrated': bool(arm.get('imus')), 'session': arm.get('session'),
                         'warnings': arm.get('warnings') or []},
            'joints': joints,
            'slew': {'calibrated': bool(y.get('left') and y.get('right')), 'sign_verified': bool(sl.get('sign_verified', True)),
                     'sign': sl.get('sign'), 'yaw_deg_s': y or None, 'onset_s': sl.get('onset_s'),
                     'session': (cal.get('slew') or {}).get('session')}}


def summary_line(m: dict) -> dict:
    """'Calibrated for: arm IMUs ✓ (session ...), ranges ✓, speeds ✓, base turn ✓, pin signs ✓'."""
    ok = lambda b: '✓' if b else '✗'                               # noqa: E731
    J = m['joints']
    arm = m['arm_imus']['calibrated']
    ranges = all(J[j]['range_deg'] for j in CAL_JOINTS)
    missing = [f'{j}-{d}' for j in CAL_JOINTS for d in DIR_NAMES[j].values() if J[j]['speed_deg_s'].get(d) is None]
    unverified = [j for j in CAL_JOINTS if not J[j]['sign_verified']]
    slew = m['slew']['calibrated'] and m['slew']['sign_verified']
    parts = [f'arm IMUs {ok(arm)}' + (f" (session {m['arm_imus']['session']})" if arm and m['arm_imus']['session'] else ''),
             f'ranges {ok(ranges)}', 'speeds ' + (f"✗ ({', '.join(missing)} missing)" if missing else '✓'),
             f'base turn {ok(slew)}', 'pin signs ' + (f"✗ ({', '.join(unverified)} not verified)" if unverified else '✓')]
    return {'text': 'Calibrated for: ' + ', '.join(parts), 'arm_imus': arm, 'ranges': ranges, 'speeds_missing': missing,
            'slew': slew, 'unverified_signs': unverified, 'ready_for_live_policy': bool(arm and ranges and not missing),
            'calibrated_at': fmt_stamp(m['arm_imus']['session']) if arm else None}


def runs(root) -> list[Path]:
    """The run directories under ``root``, oldest first (by stamp; '-sim' runs included)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if d.is_dir() and RUN_RE.match(d.name)), key=lambda d: d.name.removesuffix('-sim'))


def run_summary(d: Path, map_session=None) -> dict:
    """One run's calibration.json: status, failures, measured values; ``in_map``: it is the map's arm_imus session."""
    r = load_json(d / 'calibration.json') or {}
    cal = r.get('calibration') or {}
    sl = cal.get('slew') or {}
    stamp = d.name.removesuffix('-sim')
    return {'name': d.name, 'stamp': stamp, 'at_text': fmt_stamp(stamp), 'mode': r.get('mode', 'sim' if d.name != stamp else 'live'),
            'status': r.get('status', 'unknown'), 'error': r.get('error'),
            'failures': cal.get('failures') or [], 'warnings': cal.get('warnings') or [],
            'ranges_deg': {j: _deg(v) for j, v in (cal.get('ranges_rad') or {}).items()},
            'speeds_deg_s': {j: {DIR_NAMES[j].get(k, k): _deg(v) for k, v in s.items()} for j, s in (cal.get('speeds_rad_s') or {}).items()},
            'dead_time_s': {j: {DIR_NAMES[j].get(k, k): round(v, 3) for k, v in s.items()} for j, s in (cal.get('dead_time_s') or {}).items()},
            'signs': {j: s.get('sign') for j, s in (cal.get('signs') or {}).items()},
            'slew': ({w: sl['slew'][w].get('yaw_rate_deg_s') for w in ('left', 'right')} | {'sign': sl.get('sign')}
                     if sl.get('slew') else {'error': sl.get('error')} if sl else None),
            'in_map': bool(map_session) and stamp == map_session and d.name == stamp,
            'artifact': _rel(d / 'calibration.json')}


def last_run(root, sim: bool = False, map_session=None) -> dict | None:
    ds = [d for d in runs(root) if d.name.endswith('-sim') == sim and (d / 'calibration.json').is_file()]
    return run_summary(ds[-1], map_session) if ds else None


def calibration_status(hardware_map, root=CAL_ROOT) -> dict:
    """Everything the calibration panel shows: the map's calibration, the last live and simulated runs."""
    m = map_status(hardware_map)
    return {'summary': summary_line(m), 'map': m, 'last': last_run(root, False, m['arm_imus']['session']),
            'last_sim': last_run(root, True), 'manual_prerequisites': MANUAL_PREREQS, 'motion': MOTION,
            'root': _rel(Path(root))}


def cal_report(root=CAL_ROOT) -> dict:
    """The newest run's log (live or simulated) with its summary."""
    ds = [d for d in runs(root) if (d / 'log.txt').is_file()]
    if not ds:
        return {'text': None, 'error': 'no calibration run yet'}
    return {'text': (ds[-1] / 'log.txt').read_text(), 'run': run_summary(ds[-1])}
