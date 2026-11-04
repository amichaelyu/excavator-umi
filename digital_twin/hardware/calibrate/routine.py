"""The routine on a rig (hardware.calibrate.rig): at-rest windows, gyro classification and end-to-end drives,
closed-loop servoing (also the policy runner's start pose), the base turn, the fit, and :func:`run_calibration`."""
from __future__ import annotations

import math

import numpy as np

from digital_twin.sim import pins

from ..state_estimator import ARM_LINKS, LINK_PARENT, ArmIMUEstimator, joints_from_pitches, link_rotation, unit
from .rig import CalibrationError, joint_pins, utc_stamp

DESCENDANTS = {'boom': ('boom', 'stick', 'bucket'), 'stick': ('stick', 'bucket'), 'bucket': ('bucket',)}
TRIP_ORDER = ('bucket', 'stick', 'boom')          # the boom last: it swings the whole arm
WINDOW_S = 0.25                                   # gyro classification window
MOVING_DPS = 15.0                                 # |mean gyro vector| above this = the joint is moving
STILL_DPS = 3.0                                   # gyro within this of its mean (and of the bias) for STILL_S = at rest
STILL_S = 0.5
MIN_TRAVEL_DEG = 5.0                              # a drive that moved its link less than this did not move it
ASSIGN_TOL_DEG = 3.0                              # an IMU before the moving joint must stay within this
RIGID_TOL_DEG = 3.0                               # IMUs past it must follow it within this (+5 %)
SLACK_MAX_DEG = 10.0                              # ... unless the miss repeats within SLACK_REPEAT_DEG on every visit:
SLACK_REPEAT_DEG = 1.0                            # then it is joint slack under the changing load, not a loose IMU
SLEW_MS = (300, 600)                              # base turn: left / right presses, each undone by the other pin
SLEW_BACK_DEG = 5.0                               # ... then one press back when the base ends further than this off
SMOOTH_S = 0.1                                    # rate smoothing for the press-to-motion onset


def deg(x) -> float:
    return round(math.degrees(float(x)), 3)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


# ---------------------------------------------------------------------------- measurements
def imu(io, link, t0, t1):
    """``io.samples`` with the IMU's gyro bias (measured at rest by run_calibration) removed."""
    t, a, g = io.samples(link, t0, t1)
    b = getattr(io, 'gyro_bias', {}).get(link)
    return t, a, (g - b if b is not None and len(g) else g)


def window(io, duration_s=0.6, period_s=0.02, links=ARM_LINKS) -> dict:
    """At-rest average per IMU from status polling: {link: {'up', 'accel', 'norm_g', 'n', 'spread_deg'}}."""
    samples = {n: [] for n in links}
    seen = {n: set() for n in links}
    t_end = io.now() + duration_s
    while True:
        for n, r in io.read().items():
            if n in samples and r[2] not in seen[n]:
                samples[n].append(np.asarray(r[0], float)); seen[n].add(r[2])
        if io.now() >= t_end:
            break
        io.sleep(period_s)
    out = {}
    for n, xs in samples.items():
        if len(xs) < 3:
            raise CalibrationError(f'the {n} IMU is not live (got {len(xs)} readings); check the Pi dashboard /api/status imus')
        a = np.mean(xs, 0)
        up = unit(a)
        spread = max(math.degrees(math.acos(float(np.clip(np.dot(unit(x), up), -1, 1)))) for x in xs)
        out[n] = {'up': up, 'accel': a, 'norm_g': float(np.linalg.norm(a)), 'n': len(xs), 'spread_deg': spread}
    return out


def classify(t, gyro, window_s=WINDOW_S, spans=False):
    """'moving' / 'chatter' / 'quiet' per ``window_s`` window of one IMU's gyro [deg/s]; with ``spans``
    [(state, t_start, t_end)]."""
    out = []
    if len(t) == 0:
        return out
    edges = np.arange(t[0], t[-1] + 1e-9, window_s)
    for a in edges:
        g = gyro[(t >= a) & (t < a + window_s)]
        if len(g) < 3:
            continue
        mean = float(np.linalg.norm(g.mean(0)))
        spread = float(np.linalg.norm(g.std(0)))
        state = 'moving' if mean > MOVING_DPS else 'chatter' if spread > 10.0 else 'quiet'
        out.append((state, float(a), float(a + window_s)) if spans else state)
    return out


def moving_spans(io, d) -> list[tuple[float, float]]:
    """The windows of a drive in which its joint's own IMU was moving (chatter excluded)."""
    ts, _, g = imu(io, d['joint'], d['t0'], d['t_press_end'])
    return [(a, b) for s, a, b in classify(ts, g, spans=True) if s == 'moving']


def in_spans(t, spans):
    m = np.zeros(len(t), bool)
    for a, b in spans:
        m |= (t >= a) & (t < b)
    return m


def smooth(t, x, s=SMOOTH_S):
    """Centred moving average of ``x`` over ``s`` seconds."""
    if len(t) < 2:
        return np.asarray(x, float)
    k = max(1, int(round(s / float(np.median(np.diff(t))))))
    return np.convolve(x, np.ones(k) / k, mode='same')


def onset(t, rate, t0, v) -> float | None:
    """Press to motion [s]: from ``t0`` (the press) until the smoothed |rate| first reaches half of |v| (the steady
    rate, same units). For a ramp this is where the steady line starts, so ``duration = dq / v + onset``."""
    i = np.nonzero((t >= t0) & (np.abs(smooth(t, rate)) >= 0.5 * abs(v)))[0]
    return float(t[i[0]] - t0) if len(i) else None


def wait_still(io, t_from, timeout_s=20.0) -> tuple[float, dict]:
    """Wait until every IMU has been still for ``STILL_S`` (motor run-on and chatter over). Returns the
    time the still window ends and {link: {'up', 'accel', 'norm_g', 'spread_deg', 'n'}} over it."""
    t_end = t_from + timeout_s
    t = t_from + STILL_S
    while t <= t_end:
        io.sleep(max(0.0, t - io.now()) + 0.05)
        ok, w = True, {}
        for n in ARM_LINKS:
            ts, acc, gyr = imu(io, n, t - STILL_S, t)
            biased = n not in getattr(io, 'gyro_bias', {})
            if (len(ts) < 10 or np.abs(gyr - gyr.mean(0)).max() > STILL_DPS
                    or (not biased and np.linalg.norm(gyr.mean(0)) > STILL_DPS)
                    or np.linalg.norm(acc, axis=1).std() > 0.02):
                ok = False
                break
            a = acc.mean(0)
            up = unit(a)
            spread = max(math.degrees(math.acos(float(np.clip(np.dot(unit(x), up), -1, 1)))) for x in acc)
            w[n] = {'up': up, 'accel': a, 'norm_g': float(np.linalg.norm(a)), 'spread_deg': spread, 'n': len(ts),
                    'gyro_mean': gyr.mean(0)}
        if ok:
            return t, w
        t += 0.1
    raise CalibrationError(f'the arm did not come to rest within {timeout_s:.0f} s after a move (still chattering?)')


def drive(io, joint, direction, max_s, chunk_ms=500, retries=2, log=print) -> dict:
    """Press ``joint`` in ``direction`` in ``chunk_ms`` chunks until its own IMU shows the end of travel
    (motion, then two non-moving windows), no motion at all, or ``max_s``; then wait until every IMU
    is still. Returns the press interval, the windows, and the rest reading at the end. A drive with
    no motion *and* no chatter (a joint already at its end chatters) is a lost command (the ESP32
    dropped off Bluetooth): it is pressed again up to ``retries`` times."""
    for attempt in range(retries + 1):
        r = _drive_once(io, joint, direction, max_s, chunk_ms, log)
        if r['result'] != 'no_motion' or 'chatter' in r['states'] or attempt == retries:
            return r
        log(f'  {joint} {direction:+d}: no response at all (command lost?); pressing again')
    return r


def _drive_once(io, joint, direction, max_s, chunk_ms, log) -> dict:
    t0 = io.now()
    states, pressed, moved = [], 0.0, False
    result = 'max_time'
    while pressed < max_s:
        a = io.now()
        io.press(joint, direction, chunk_ms)
        pressed += chunk_ms / 1000.0
        ts, _, gyr = imu(io, joint, a, io.now())
        s = classify(ts, gyr)
        states += s
        moved = moved or 'moving' in s
        if moved and len(states) >= 2 and states[-1] != 'moving' and states[-2] != 'moving':
            result = 'end'
            break
        if not moved and pressed >= 1.5:
            result = 'no_motion'
            break
    t_press_end = io.now()
    others = [n for n in ARM_LINKS if n != joint and 'moving' in classify(*imu(io, n, t0, t_press_end)[::2])]
    t_rest, rest = wait_still(io, t_press_end)
    log(f'  {joint} {direction:+d}: {result.replace("_", " ")} after {pressed:.0f} s of presses '
        f'({states.count("moving")} moving / {states.count("chatter")} chatter windows), still after '
        f'{t_rest - t_press_end:.1f} s')
    return {'joint': joint, 'direction': int(direction), 'result': result, 'moved': moved, 'others_moved': others, 't0': t0,
            't_press_end': t_press_end, 't_rest': t_rest, 'pressed_s': pressed, 'states': states, 'rest': rest}


def measure_joints(io, est: ArmIMUEstimator, duration_s=0.4) -> dict:
    """Static joint angles {joint: rad} from one at-rest window through the calibration (no filter)."""
    w = window(io, duration_s)
    return joints_from_pitches({n: est.raw_pitch(n, w[n]['accel']) for n in ARM_LINKS})


def servo_joint(io, est, joint, target, speeds, tol_rad=math.radians(1.5), max_iter=14, settle_s=0.3,
                min_ms=60, max_ms=2000, log=print) -> dict:
    """Closed-loop: pulse ``joint`` toward ``target`` (rad) by the measured error until within ``tol_rad``
    (or within 0.6 of the motion of one ``min_ms`` pulse, the finest step the joint can make).
    ``speeds`` = (v_pos, v_neg) rad/s, adapted from each pulse's measured motion."""
    v = {1: float(speeds[0]), -1: float(speeds[1])}
    q = measure_joints(io, est)[joint]
    hist, stuck = [], 0
    for _ in range(max_iter):
        err = target - q
        d = 1 if err > 0 else -1
        if abs(err) <= max(tol_rad, 0.6 * v[d] * min_ms / 1000.0):    # finer than one minimum pulse can go
            break
        ms = int(np.clip(abs(err) / v[d] * 1000 * 0.85, min_ms, max_ms))
        io.pulse(joint, d, ms)
        io.sleep(settle_s)
        q1 = measure_joints(io, est)[joint]
        moved = (q1 - q) * d
        hist.append({'ms': ms, 'dir': d, 'moved_deg': deg(moved), 'q_deg': deg(q1)})
        if moved < -math.radians(1.0):
            raise CalibrationError(f'{joint}: a {d:+d} pulse moved it {math.degrees(moved):+.1f} deg the wrong way; '
                                   'check the pin signs and the IMU calibration')
        if moved < math.radians(0.2):
            stuck += 1
            if stuck >= 2:
                raise CalibrationError(f'{joint}: no motion for two {d:+d} pulses at {math.degrees(q1):.1f} deg '
                                       f'(target {math.degrees(target):.1f} deg; at the end or stalled?)')
        else:
            stuck = 0
            v[d] = 0.5 * v[d] + 0.5 * moved / (ms / 1000.0)
        q = q1
    else:
        d = 1 if target > q else -1
        if abs(target - q) > max(tol_rad, 0.6 * v[d] * min_ms / 1000.0):
            raise CalibrationError(f'{joint}: {math.degrees(target - q):+.1f} deg from the target after {max_iter} pulses')
    log(f'  {joint}: {math.degrees(q):+.1f} deg (target {math.degrees(target):+.1f}) after {len(hist)} pulses')
    return {'final_rad': float(q), 'target_rad': float(target), 'pulses': hist}


# ---------------------------------------------------------------------------- base turn
def turn(io, direction, ms, up, log=print) -> dict:
    """One base-turn press, then rest. Per link IMU its yaw rate = gyro . its at-rest up reading ``up[link]`` (+ =
    counter-clockwise seen from above; the arm is still, so up is fixed in each IMU frame): the total turn, the
    steady rate (smoothed rate within half of its peak) and the press-to-motion onset; the medians over the IMUs."""
    t0 = io.now()
    io.press('slew', direction, ms)
    t_rest, _ = wait_still(io, io.now())
    per = []
    for n in ARM_LINKS:
        t, _, g = imu(io, n, t0 - 0.3, t_rest)
        if len(t) < 10:
            continue
        y = g @ up[n]
        m = t >= t0
        ys = smooth(t, y)
        peak = float(np.max(np.abs(ys[m]))) if m.any() else 0.0
        rate = float(np.median(ys[m & (np.abs(ys) >= 0.5 * peak)])) if peak >= MOVING_DPS else None
        per.append({'yaw_deg': float(np.trapezoid(y[m], t[m])), 'rate': rate,
                    'onset': None if rate is None else onset(t, y, t0, rate)})
    r = {'direction': int(direction), 'ms': int(ms), 'yaw_deg': _median(p['yaw_deg'] for p in per),
         'rate_deg_s': _median(p['rate'] for p in per), 'onset_s': _median(p['onset'] for p in per)}
    log(f"  base {'left' if direction > 0 else 'right'} {ms} ms: "
        + ('no turn' if r['rate_deg_s'] is None else
           f"{r['yaw_deg']:+.1f} deg, {r['rate_deg_s']:+.1f} deg/s, onset {r['onset_s']:.2f} s"))
    return r


def measure_slew(io, hw, log=print) -> dict:
    """The base turn from the link IMUs, starting with the base facing forward (slew 0): per SLEW_MS a left press
    undone by a right one, then a right press undone by a left one (the run-on differs, so one more press turns it
    back when it ends more than SLEW_BACK_DEG off). Per direction (left = the positive pin) the steady turn rate, the
    mean rate over the press and the press-to-motion onset; the pin sign is +1 when the left pin turns the base
    counter-clockwise seen from above (verified only then)."""
    if not io.operator('confirm', 'Next: the BASE turns left and right and comes back. Is the base facing forward '
                                  '(slew 0) with room to turn?'):
        raise CalibrationError('base turn not confirmed; the base was not moved')
    _, rest = wait_still(io, io.now())
    up = {n: rest[n]['up'] for n in ARM_LINKS}
    segs = []
    for ms in SLEW_MS:
        for d in (1, -1):
            segs += [turn(io, d, ms, up, log), turn(io, -d, ms, up, log)]
    sl = joint_pins(hw, 'slew')
    out = {}
    for d, w in ((1, 'left'), (-1, 'right')):
        s = [x for x in segs if x['direction'] == d]
        out[w] = {'pin': pins.entry_pin(sl, d), 'n_pulses': len(s),
                  'yaw_rate_deg_s': _median(x['rate_deg_s'] for x in s), 'onset_s': _median(x['onset_s'] for x in s),
                  'mean_yaw_deg_s': _median(None if x['yaw_deg'] is None else x['yaw_deg'] / (x['ms'] / 1000) for x in s),
                  'pulses': [{'ms': x['ms'], 'yaw_deg': x['yaw_deg']} for x in s]}
    left, right = out['left']['yaw_rate_deg_s'], out['right']['yaw_rate_deg_s']
    moved = bool(left and right and min(abs(left), abs(right)) >= 3.0)
    sign = (1 if left > 0 > right else -1 if left < 0 < right else 0) if moved else 0
    net = float(sum(x['yaw_deg'] or 0.0 for x in segs))
    back = None
    if sign and abs(net) > SLEW_BACK_DEG:
        d = -sign if net > 0 else sign                     # the pin that turns it the other way
        v = abs(out['left' if d > 0 else 'right']['yaw_rate_deg_s'])
        log(f'  base {net:+.1f} deg off forward: turning it back')
        back = turn(io, d, int(np.clip(abs(net) / v * 1000, 50, 3000)), up, log)
        net += back['yaw_deg'] or 0.0
    return {'slew': out, 'sign': sign, 'sign_verified': sign == 1, 'back_press': back,
            'reason': ('the left pin turns the base counter-clockwise (seen from above)' if sign > 0 else
                       'pins reversed: the left pin turns the base clockwise' if sign < 0 else
                       'no clear opposite turn from the two pins (base not moving?)'),
            'net_yaw_deg': net, 'start': 'facing forward (slew 0)',
            'yaw_sign_convention': '+ = counter-clockwise seen from above (right-hand rotation about up)'}


# ---------------------------------------------------------------------------- fit
def gyro_axis(io, drives, link) -> tuple[np.ndarray | None, int]:
    """``link``'s joint axis in its IMU frame: the principal direction of its gyro while a joint on or
    before it moved, oriented so a + drive turns it +. Prefers its own joint's drives."""
    for own in (True, False):
        rows, signs = [], []
        for d in drives:
            if not d['moved'] or link not in DESCENDANTS[d['joint']] or (own and d['joint'] != link):
                continue
            ts, _, g = imu(io, link, d['t0'], d['t_press_end'])
            g = g[in_spans(ts, moving_spans(io, d)) & (np.linalg.norm(g, axis=1) > MOVING_DPS)]
            if len(g) >= 5:
                rows.append(g); signs.append(d['direction'])
        if rows:
            G = np.vstack(rows)
            axis = np.linalg.svd(G, full_matrices=False)[2][0]
            score = sum(s * float(np.mean(r @ axis)) for r, s in zip(rows, signs))
            return (axis if score > 0 else -axis), len(G)
    return None, 0


def drive_speed(io, d, axis) -> float | None:
    """Median |rate| [rad/s] about ``axis`` of the joint's own IMU while it was moving (parent still)."""
    if axis is None or not d['moved']:
        return None
    ts, _, g = imu(io, d['joint'], d['t0'], d['t_press_end'])
    r = np.abs(g[in_spans(ts, moving_spans(io, d))] @ axis)
    r = r[r > MOVING_DPS]
    return math.radians(float(np.median(r))) if len(r) >= 5 else None


def drive_onset(io, d, axis, v) -> float | None:
    """Press to motion [s] of a drive that moved at ``v`` rad/s: its first press to its own IMU's rate about ``axis``."""
    if axis is None or v is None:
        return None
    ts, _, g = imu(io, d['joint'], d['t0'], d['t_press_end'])
    return onset(ts, g @ axis, d['t0'], math.degrees(v)) if len(ts) >= 10 else None


def fit_calibration(io, ref: list[dict], drives: list[dict], trips: dict, hw: dict) -> dict:
    """Axes, reference, ranges, speeds, dead times, pin signs and checks. ``ref``: rest readings with every joint
    at its home end; ``trips``: {joint: [(away drive, home drive), ...]}."""
    homing = hw['homing']['joints']
    home_dir = {j: int(homing[j]['home_direction']) for j in ARM_LINKS}
    failures, warnings, checks = [], [], {'joints': {}}
    axes = {}
    for n in ARM_LINKS:
        a, k = gyro_axis(io, drives, n)
        if a is None:
            raise CalibrationError(f'no drive turned the {n} IMU; cannot find its axis (no pin moved the {n} or the arm)')
        axes[n] = a
        checks.setdefault('axis_samples', {})[n] = k
    up_ref = {}
    for n in ARM_LINKS:
        ups = np.array([r[n]['up'] for r in ref])
        up_ref[n] = unit(ups.mean(0))
        checks.setdefault('reference_spread_deg', {})[n] = round(max(
            abs(math.degrees(link_rotation(up_ref[n], u, axes[n]))) for u in ups), 3)
    ranges, speeds, onsets, reversed_ = {}, {}, {}, set()
    for j in ARM_LINKS:
        c = checks['joints'][j] = {'drives': {}, 'visits': []}   # per direction (the pin: joints.<j>.<direction>)
        jm = joint_pins(hw, j)
        for d in [d for d in drives if d['joint'] == j]:
            p = c['drives'].setdefault(pins.side(d['direction']), {'moved': False, 'results': []})
            p['moved'] = p['moved'] or d['moved']
            p['results'].append(d['result'])
            v = drive_speed(io, d, axes[j])
            if v is not None:
                k = 'positive' if d['direction'] > 0 else 'negative'
                speeds.setdefault(j, {}).setdefault(k, []).append(v)
                o = drive_onset(io, d, axes[j], v)
                if o is not None:
                    onsets.setdefault(j, {}).setdefault(k, []).append(o)
        for k, p in c['drives'].items():
            if not p['moved']:
                failures.append(f'GPIO {pins.entry_pin(jm, k)} ({j} {k}) did not move the {j} ({", ".join(p["results"])})')
        for d in [d for d in drives if d['joint'] == j and not d['moved'] and d['others_moved']]:
            failures.append(f'pressing the {j} {d["direction"]:+d} moved the {", ".join(d["others_moved"])} IMU but not the '
                            f'{j} IMU: the IMUs are not on the links imus.json names (swapped bus/wiring?)')
        par = LINK_PARENT[j]
        dqs = []
        misses = {}
        for away, home in trips.get(j, []):
            if not away['moved']:
                continue
            rot = {n: link_rotation(home['rest'][n]['up'], away['rest'][n]['up'], axes[n]) for n in ARM_LINKS}
            back = {n: link_rotation(ref[0][n]['up'], home['rest'][n]['up'], axes[n]) for n in ARM_LINKS}
            dq = rot[j] - (rot[par] if par else 0.0)
            dqs.append(dq)
            visit = {'range_deg': deg(abs(dq)), 'link_rotation_deg': {n: deg(v) for n, v in rot.items()},
                     'home_repeat_deg': {n: deg(v) for n, v in back.items()}}
            c['visits'].append(visit)
            if dq * -home_dir[j] <= 0:
                failures.append(f'{j}: its away drive turned it {math.degrees(dq):+.1f} deg, expected sign {-home_dir[j]:+d} '
                                '(pins reversed or IMUs swapped)')
            for n in ARM_LINKS:
                if n not in DESCENDANTS[j] and abs(math.degrees(rot[n])) > ASSIGN_TOL_DEG:
                    failures.append(f'moving the {j} turned the {n} IMU by {math.degrees(rot[n]):+.1f} deg: that IMU is '
                                    f'not on the {n} (swapped bus/wiring in imus.json?) or the base moved')
            for n in DESCENDANTS[j][1:]:
                dd = math.degrees(rot[n] - rot[j])
                tol = RIGID_TOL_DEG + 0.05 * abs(math.degrees(rot[j]))
                if abs(math.degrees(rot[j])) > MIN_TRAVEL_DEG and abs(math.degrees(rot[n] + rot[j])) <= tol:
                    reversed_.add((j, n))
                    failures.append(f'moving the {j} turned the {n} IMU the opposite way ({math.degrees(rot[n]):+.1f} vs '
                                    f'{math.degrees(rot[j]):+.1f} deg): the {n} pins are reversed relative to the {j} pins '
                                    f'(swap the positive / negative pins of the {n} in hardware_map.json)')
                elif abs(dd) > tol:
                    misses.setdefault(n, []).append(dd)
        for n, dds in misses.items():
            if len(dds) > 1 and np.ptp(dds) <= SLACK_REPEAT_DEG and max(abs(x) for x in dds) <= SLACK_MAX_DEG:
                c.setdefault('slack_deg', {})[n] = round(float(np.mean(dds)), 2)
                warnings.append(f'moving the {j} turned the {n} IMU {np.mean(dds):+.1f} deg differently from the {j} IMU, '
                                f'the same on every visit: slack in the {n} joint under the changing load (accepted)')
            else:
                failures += [f'moving the {j} turned the {n} IMU {dd:+.1f} deg differently from the {j} IMU '
                             '(loose mount or the wrong link)' for dd in dds]
        if dqs:
            ranges[j] = float(abs(np.mean(dqs)))
            c['range_spread_deg'] = round(math.degrees(float(np.ptp(np.abs(dqs)))), 3)
    ref_q = {j: float(homing[j]['home_value_rad']) for j in ARM_LINKS}
    stops = {j: ([ref_q[j] - r, ref_q[j]] if home_dir[j] > 0 else [ref_q[j], ref_q[j] + r]) for j, r in ranges.items()}
    return {'imus': {n: {'axis_imu': axes[n].tolist(), 'up_ref_imu': up_ref[n].tolist()} for n in ARM_LINKS},
            'reference_q_rad': ref_q, 'reference_pose': 'every joint at its home end (stick up, boom up, bucket in)',
            'ranges_rad': ranges, 'stops_rad': stops,
            'speeds_rad_s': {j: {k: float(np.median(v)) for k, v in s.items()} for j, s in speeds.items()},
            'dead_time_s': {j: {k: float(np.median(v)) for k, v in s.items()} for j, s in onsets.items()},
            'signs': pin_signs(checks, reversed_),
            'checks': checks, 'failures': failures, 'warnings': warnings,
            'convention': 'link pitch = pitch_ref + rotation of the at-rest gravity reading about axis_imu (right-handed, '
                          '+ = the joint\'s model + direction); boom = boom pitch, stick = stick - boom, bucket = bucket - stick'}


def pin_signs(checks: dict, reversed_: set) -> dict:
    """Per joint whether its pins agree with the others'. The axes are oriented by each joint's own + drives, so
    the link IMUs see only relative reversals: a child IMU turning the opposite way while its parent moves.
    The joint in the most reversed pairs (with one pair: the child) is the reversed one (sign -1); a joint
    both of whose pins moved it and that is in no reversed pair is +1 (verified); otherwise unknown."""
    count = {}
    for pair in reversed_:
        for x in pair:
            count[x] = count.get(x, 0) + 1
    blame = {x for x, k in count.items() if k >= 2} or {n for _, n in reversed_}
    out = {}
    for j in ARM_LINKS:
        drives = checks['joints'][j]['drives']
        others = sorted({x for pair in reversed_ if j in pair for x in pair} - {j})
        if j in blame:
            out[j] = {'sign': -1, 'verified': True, 'reason': f'reversed relative to the {", ".join(others)} pins'}
        elif len(drives) == 2 and all(p['moved'] for p in drives.values()):
            out[j] = {'sign': 1, 'verified': True, 'reason': 'both pins move it; its IMU turns with the other links'}
        else:
            out[j] = {'sign': None, 'verified': False, 'reason': 'a pin did not move it'}
    return out


# ---------------------------------------------------------------------------- routine
def run_calibration(io, hw: dict, args, log=print) -> dict:
    homing = hw['homing']['joints']
    home_dir = {j: int(homing[j]['home_direction']) for j in ARM_LINKS}
    from ..state_estimator import load_joint_config
    prior = load_joint_config()
    max_s = {j: (prior[j]['hi'] - prior[j]['lo']) / min(prior[j]['v_pos'], prior[j]['v_neg']) * 1.5 + 2.0 for j in ARM_LINKS}
    out = {'mode': args.mode, 'started': utc_stamp(), 'visits': args.visits}
    drives = []

    def go(j, d):
        r = drive(io, j, d, max_s[j], log=log)
        drives.append(r)
        return r

    log('1/6 check: streaming the boom, stick and bucket IMUs')
    t0 = io.now()
    io.sleep(1.0)
    for n in ARM_LINKS:
        if len(io.samples(n, t0, io.now())[0]) < 10:
            raise CalibrationError(f'the {n} IMU is not live (no samples in 1 s); check the Pi dashboard /api/status imus')
    t, w = wait_still(io, io.now(), timeout_s=10.0)
    io.gyro_bias = {n: w[n]['gyro_mean'] for n in ARM_LINKS}
    out['gyro_bias_dps'] = {n: [round(float(x), 3) for x in b] for n, b in io.gyro_bias.items()}
    for n in ARM_LINKS:
        log(f'  {n}: |a| {w[n]["norm_g"]:.3f} g, {w[n]["n"]} samples in {STILL_S:.1f} s, spread {w[n]["spread_deg"]:.2f} deg, '
            f'gyro bias {np.round(io.gyro_bias[n], 2).tolist()} deg/s')
        if not 0.8 <= w[n]['norm_g'] <= 1.2:
            raise CalibrationError(f'the {n} IMU reads {w[n]["norm_g"]:.2f} g at rest (expected ~1 g); check it')

    log('2/6 home: stick up, boom up, bucket in')
    for j in ('stick', 'boom', 'bucket'):
        go(j, home_dir[j])
    ref = [drives[-1]['rest']]
    log('  reference pose recorded (every joint at its home end)')

    log(f'3/6 round trips: {args.visits} per joint (away end, then home)')
    trips = {j: [] for j in ARM_LINKS}
    for j in TRIP_ORDER:
        if j == 'boom':
            if not io.operator('confirm', 'Next: the stick and bucket go OUT, then the BOOM goes all the way DOWN. Is the '
                                          'arm clear of the ground and everything else at full boom down, arm out?'):
                raise CalibrationError('boom-down not confirmed; nothing more was moved')
            log('  stick and bucket out for the boom trips')
            go('stick', -home_dir['stick'])
            go('bucket', -home_dir['bucket'])
        for _ in range(args.visits):
            away = go(j, -home_dir[j])
            home = go(j, home_dir[j])
            trips[j].append((away, home))
            if j != 'boom':
                ref.append(home['rest'])                          # every joint home
            if not away['moved']:
                break                                              # a dead pin: one try is enough
    log('  home again: stick up, bucket in')
    go('stick', home_dir['stick'])
    ref.append(go('bucket', home_dir['bucket'])['rest'])
    out['drives'] = [{k: v for k, v in d.items() if k != 'rest'} for d in drives]

    slew = None
    if getattr(args, 'slew', True):
        log('4/6 base turn: left and right from facing forward, each press undone by the other pin')
        try:
            slew = measure_slew(io, hw, log)
            log(f"  base turn: left {slew['slew']['left']['yaw_rate_deg_s'] or 0:+.1f} deg/s, right "
                f"{slew['slew']['right']['yaw_rate_deg_s'] or 0:+.1f} deg/s, sign {slew['sign']:+d} ({slew['reason']}), "
                f"net {slew['net_yaw_deg']:+.1f} deg")
        except CalibrationError as ex:
            log(f'  base turn skipped: {ex}')
            slew = {'error': str(ex)}
    else:
        log('4/6 base turn: skipped (--no-slew)')

    log('5/6 fit: axes from the gyro, reference, ranges, speeds, dead times, pin signs')
    cal = fit_calibration(io, ref, drives, trips, hw)
    cal['gyro_bias_dps'] = out['gyro_bias_dps']                 # at rest, step 1; ArmIMUEstimator subtracts it
    cal['slew'] = slew
    for j in ARM_LINKS:
        r = cal['ranges_rad'].get(j)
        sp, dt = cal['speeds_rad_s'].get(j, {}), cal['dead_time_s'].get(j, {})
        log(f'  {j}: range ' + (f'{math.degrees(r):.1f} deg (spread {cal["checks"]["joints"][j]["range_spread_deg"]:.1f})'
                                if r is not None else 'not measured')
            + ', speed ' + ', '.join(f'{k[:3]} {math.degrees(v):.0f} deg/s' for k, v in sp.items())
            + ', onset ' + ', '.join(f'{k[:3]} {v:.2f} s' for k, v in dt.items())
            + f', reference spread {cal["checks"]["reference_spread_deg"][j]:.2f} deg'
            + f', pins {cal["signs"][j]["sign"] or "?"} ({cal["signs"][j]["reason"]})')
    out['calibration'] = cal
    hard = [f for f in cal['failures'] if not f.startswith('GPIO ')]
    for f in cal['failures']:
        log(f'  CHECK FAILED: {f}')
    for w in cal.get('warnings', []):
        log(f'  NOTE: {w}')
    if hard and not args.force:
        out['status'] = 'failed checks'
        return out
    log('6/6 done' + (' (pins that did not move are listed above; their joints have no range)' if cal['failures'] else ''))
    out['status'] = 'done'
    return out


class _NullBase:
    """Base estimator stand-in for static measurement during calibration."""
    joints = ARM_LINKS


def watch(io, hw: dict, period_s=0.5, log=print) -> None:
    """Print the joint angles the calibration gives from at-rest readings, until Ctrl-C."""
    est = ArmIMUEstimator.from_hardware_map(_NullBase(), hw)
    while True:
        q = measure_joints(io, est, 0.3)
        log('  '.join(f'{j} {math.degrees(v):+7.1f} deg' for j, v in q.items()))
        io.sleep(period_s)
