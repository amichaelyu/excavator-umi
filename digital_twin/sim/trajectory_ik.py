"""Offline IK with path-wide braking lookahead and local time scaling.

The geometric path and the clock are separate. A backward reachability pass
finds speeds from which all upcoming limits and the final stop remain feasible;
a forward pass chooses the fastest feasible progress, never faster than the
recording. This is path timing, not a model of soil forces or motor control.

Joint paths use shape-preserving cubic interpolation. Velocity and acceleration
extrema are checked over every polynomial interval, including between control
ticks. Exported source times retain the correspondence to demonstration images.
The robot is assumed to start at the first pose, at rest, and finishes at rest.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, linprog


@dataclass
class TimedPath:
    time: np.ndarray
    source_time: np.ndarray
    q: np.ndarray
    path: PchipInterpolator
    metrics: dict


def _vector(value, n, name, *, positive=False):
    value = np.broadcast_to(np.asarray(value, float), (n,)).copy()
    if not np.isfinite(value).all() or (positive and np.any(value <= 0)):
        raise ValueError(f'{name} must contain finite' + (' positive values' if positive else ' values'))
    return value


def _extrema(poly):
    """Endpoints and real stationary points of an ascending polynomial on [0,1]."""
    derivative = np.polynomial.polynomial.polyder(poly)
    roots = np.polynomial.polynomial.polyroots(np.trim_zeros(derivative, 'b')) if np.any(derivative) else []
    return np.array([0., 1., *[float(z.real) for z in roots if abs(z.imag) < 1e-8 and 0 < z.real < 1]])


def _continuous_limits(path, grid, x, v_pos, v_neg, acceleration):
    """Exact polynomial extrema for q'(s)*sqrt(x) and q''(s)*x + q'(s)*u."""
    velocity_ratio, acceleration_ratio = 0., 0.
    for i, h in enumerate(np.diff(grid)):
        segment = min(np.searchsorted(path.x, grid[i], side='right') - 1, len(path.x) - 2)
        offset = grid[i] - path.x[segment]
        c3, c2, c1, _ = path.c[:, segment]
        # r is local interval progress in [0,1], for well-conditioned roots.
        derivative = np.array([3 * c3 * offset**2 + 2 * c2 * offset + c1,
                               (6 * c3 * offset + 2 * c2) * h, 3 * c3 * h*h])
        curvature = np.array([6 * c3 * offset + 2 * c2, 6 * c3 * h])
        speed2 = np.array([x[i], x[i + 1] - x[i]])
        u = (x[i + 1] - x[i]) / (2 * h)
        for j in range(len(v_pos)):
            qp = derivative[:, j]
            squared_velocity = np.polynomial.polynomial.polymul(
                np.polynomial.polynomial.polymul(qp, qp), speed2)
            points = _extrema(squared_velocity)
            sign = np.polynomial.polynomial.polyval(points, qp)
            velocity = np.sqrt(np.maximum(0, np.polynomial.polynomial.polyval(points, squared_velocity)))
            velocity_ratio = max(velocity_ratio, float(np.max(velocity / np.where(sign >= 0, v_pos[j], v_neg[j]))))
            acc = np.polynomial.polynomial.polyadd(
                np.polynomial.polynomial.polymul(curvature[:, j], speed2), u * qp)
            acceleration_ratio = max(acceleration_ratio, float(np.max(np.abs(
                np.polynomial.polynomial.polyval(_extrema(acc), acc))) / acceleration[j]))
    return velocity_ratio, acceleration_ratio


def retime_joint_path(q, source_time, *, lower, upper, v_pos, v_neg, acceleration,
                      dt=0.05, subdivisions=2, margin=0.98):
    """Preserve a joint path, slow locally, and enforce continuous motion limits.

    ``source_time`` is strictly increasing. No waypoint is dropped or reordered.
    Returned samples are uniformly spaced by ``dt`` and include both endpoints.
    Acceleration is bounded on either side of each knot; jerk is not constrained.
    """
    before = time.monotonic()
    q, source_time = np.asarray(q, float), np.asarray(source_time, float)
    if q.ndim != 2 or len(q) < 2 or source_time.shape != (len(q),):
        raise ValueError('Expected q [N,joints] and source_time [N], N >= 2')
    if not np.isfinite(q).all() or not np.isfinite(source_time).all() or np.any(np.diff(source_time) <= 0):
        raise ValueError('Path must be finite with strictly increasing source times')
    if not np.isfinite(dt) or dt <= 0 or not 0 < margin < 1 or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError('Invalid timing options')
    n = q.shape[1]
    lower, upper = _vector(lower, n, 'lower'), _vector(upper, n, 'upper')
    v_pos, v_neg = _vector(v_pos, n, 'v_pos', positive=True), _vector(v_neg, n, 'v_neg', positive=True)
    acceleration = _vector(acceleration, n, 'acceleration', positive=True)
    if np.any(lower >= upper) or np.any(q < lower - 1e-8) or np.any(q > upper + 1e-8):
        raise ValueError('Joint path exceeds position limits')
    path = PchipInterpolator(source_time, np.clip(q, lower, upper), axis=0)
    grid = np.r_[np.concatenate([np.linspace(a, b, subdivisions, endpoint=False)
                                for a, b in zip(source_time[:-1], source_time[1:])]), source_time[-1]]
    constraints = []
    for i, h in enumerate(np.diff(grid)):
        rows, bounds = [], []
        for r in (0., .25, .5, .75, 1.):
            # Evaluate both one-sided curvatures at PCHIP knots.
            s = grid[i] + r * h
            if r == 1.:
                s = np.nextafter(s, grid[i])
            qp, qpp = path(s, 1), path(s, 2)
            ab = np.c_[qpp * (1 - r) - qp / (2 * h), qpp * r + qp / (2 * h)] / acceleration[:, None]
            rows.extend(ab); rows.extend(-ab); bounds.extend(np.full(2 * n, margin))
            moving = np.abs(qp) > 1e-12
            if moving.any():
                cap = np.min((np.where(qp[moving] >= 0, v_pos[moving], v_neg[moving]) * margin / qp[moving])**2)
                rows.append([1 - r, r]); bounds.append(min(1., float(cap)))
        constraints.append((np.asarray(rows), np.asarray(bounds)))
    # x = (ds/dt)^2. x_next = x + 2*ds*u makes each interval's constraints linear.
    # At each node the controllable speeds form [0,K]: zero is always feasible.
    controllable = np.zeros(len(grid))
    for i in range(len(grid) - 2, -1, -1):
        A, b = constraints[i]
        result = linprog([-1., 0.], A_ub=A, b_ub=b,
                         bounds=[(0., 1.), (0., controllable[i + 1])], method='highs')
        if not result.success:
            raise ValueError(f'Cannot find a feasible braking path at interval {i}: {result.message}')
        controllable[i] = max(0., min(1., result.x[0]))
    x = np.zeros(len(grid))
    for i, (A, b) in enumerate(constraints):
        remainder = b - A[:, 0] * x[i]
        positive, negative = A[:, 1] > 1e-12, A[:, 1] < -1e-12
        high = min(controllable[i + 1], float(np.min(remainder[positive] / A[positive, 1], initial=np.inf)))
        low = max(0., float(np.max(remainder[negative] / A[negative, 1], initial=-np.inf)))
        if high < low - 1e-7 or np.any(remainder[~(positive | negative)] < -1e-7):
            raise ValueError(f'Infeasible forward timing at interval {i}')
        x[i + 1] = max(0., high)
    denominator = np.sqrt(x[:-1]) + np.sqrt(x[1:])
    if np.any(denominator < 1e-12):
        raise ValueError('Timing stalled; the path needs additional intermediate samples')
    intervals = 2 * np.diff(grid) / denominator
    raw_time = np.r_[0., np.cumsum(intervals)]
    vr, ar = _continuous_limits(path, grid, x, v_pos, v_neg, acceleration)
    # Collocation locates a fast schedule; the continuous certificate corrects
    # any between-sample excess without weakening a limit or clipping a joint.
    dilation = max(1., vr / margin, np.sqrt(ar / margin))
    ticks = max(1, int(np.ceil(raw_time[-1] * dilation / dt)))
    dilation = ticks * dt / raw_time[-1]
    x /= dilation**2
    clock = raw_time * dilation
    t = np.arange(ticks + 1) * dt
    index = np.clip(np.searchsorted(clock, t, side='right') - 1, 0, len(grid) - 2)
    elapsed = t - clock[index]
    u = np.diff(x) / (2 * np.diff(grid))
    phase = grid[index] + np.sqrt(x[index]) * elapsed + .5 * u[index] * elapsed**2
    phase = np.clip(phase, grid[0], grid[-1]); phase[0], phase[-1] = grid[0], grid[-1]
    if np.any(np.diff(phase) <= 0):
        raise ValueError('Retimed source correspondence is not strictly increasing')
    sampled = path(phase)
    metrics = {'method': 'backward_reachability_forward_progress', 'lookahead': 'complete recorded path',
               'source_duration_s': float(source_time[-1] - source_time[0]), 'duration_s': float(t[-1]),
               'duration_ratio': float(t[-1] / (source_time[-1] - source_time[0])),
               'continuous_velocity_limit_ratio': vr / dilation,
               'continuous_acceleration_limit_ratio': ar / dilation**2,
               'acceleration_limits_rad_s2': acceleration.tolist(), 'max_source_speed': float(np.sqrt(x.max())),
               'start_joint_velocity': [0.] * n, 'end_joint_velocity': [0.] * n,
               'start_pose_assumption': 'already at the first path pose', 'jerk_limited': False,
               'solve_seconds': time.monotonic() - before, 'subdivisions': subdivisions}
    return TimedPath(t, phase, sampled, path, metrics)


def fit_joint_path(arm, targets, pitches, *, reference_q=None, q_init=None, locked=None, w_pitch=0.2):
    """Solve position and bucket tilt together, including hardware with held joints.

    The analytical/warm-start IK supplies branches. A bounded nonlinear solve
    improves their weighted pose error when both targets cannot be met exactly.
    Remaining pose errors are reported; retiming cannot make them reachable.
    """
    from .kinematics import JOINTS, pitch_of, solve_ik
    targets, pitches = np.asarray(targets, float), np.asarray(pitches, float)
    if targets.ndim != 2 or targets.shape[1] != 3 or pitches.shape != (len(targets),) or len(targets) < 2:
        raise ValueError('Expected targets [N,3] and pitches [N], N >= 2')
    if not np.isfinite(targets).all() or not np.isfinite(pitches).all() or not np.isfinite(w_pitch) or w_pitch <= 0:
        raise ValueError('Targets and positive pitch weight must be finite')
    locked = dict(locked or {})
    if set(locked) - set(JOINTS):
        raise ValueError('Unknown locked joint')
    free = np.array([name not in locked for name in JOINTS])
    if reference_q is None:
        reference_q, _, _ = solve_ik(arm, targets, pitches, q_init=q_init, locked=locked, w_pitch=w_pitch)
    q = np.asarray(reference_q, float).copy()
    if q.shape != (len(targets), len(JOINTS)) or not np.isfinite(q).all():
        raise ValueError('Invalid reference joint path')
    if np.any(q < arm.lo - 1e-7) or np.any(q > arm.hi + 1e-7):
        raise ValueError('Reference joint path exceeds limits')
    q = np.clip(q, arm.lo, arm.hi)
    for name, value in locked.items():
        j = JOINTS.index(name)
        if not np.isfinite(value) or not arm.lo[j] <= value <= arm.hi[j]:
            raise ValueError('Locked joint exceeds limits')
        q[:, j] = value
    for i in range(len(q)):
        def residual(values):
            state = q[i].copy(); state[free] = values
            return np.r_[arm.fk(state) - targets[i], w_pitch * (pitch_of(state) - pitches[i])]
        def jacobian(values):
            state = q[i].copy(); state[free] = values
            return np.vstack([arm.jac(state), w_pitch * arm.pitch_mask])[..., free]
        if free.any() and np.linalg.norm(residual(q[i, free])) > 1e-5:
            result = least_squares(residual, q[i, free], jac=jacobian, bounds=(arm.lo[free], arm.hi[free]),
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=80)
            if np.isfinite(result.x).all() and np.linalg.norm(result.fun) <= np.linalg.norm(residual(q[i, free])):
                q[i, free] = result.x
    return q


def plan_ik_trajectory(arm, targets, pitches, source_time, *, reference_q=None, q_init=None,
                       locked=None, w_pitch=0.2, ramp_s=0.1, dt=0.05):
    if not np.isfinite(ramp_s) or ramp_s <= 0:
        raise ValueError('Lookahead timing requires a positive acceleration ramp duration')
    q = fit_joint_path(arm, targets, pitches, reference_q=reference_q, q_init=q_init,
                       locked=locked, w_pitch=w_pitch)
    source_time = np.asarray(source_time, float)
    if source_time.shape != (len(q),) or not np.isfinite(source_time).all() or np.any(np.diff(source_time) <= 0):
        raise ValueError('Source times must be finite and strictly increasing')
    targets, pitches = np.asarray(targets, float), np.asarray(pitches, float)
    grid = source_time.copy()
    original_count = len(grid)
    from .kinematics import pitch_of

    def desired(t):
        return (np.column_stack([np.interp(t, source_time, targets[:, j]) for j in range(3)]),
                np.interp(t, source_time, pitches))

    # Smooth joints do not imply the tooth follows a straight Cartesian segment.
    # Insert IK waypoints where joint interpolation bows away from the recorded
    # path, then retime the refined curve so its motion limits still hold.
    for iteration in range(7):
        curve = PchipInterpolator(grid, q, axis=0)
        probes = (grid[:-1, None] + np.diff(grid)[:, None] * [.25, .5, .75]).ravel()
        want_p, want_phi = desired(probes)
        probe_q = curve(probes)
        position_error = np.linalg.norm(np.array([arm.fk(v) for v in probe_q]) - want_p, axis=1)
        angle_error = np.abs(pitch_of(probe_q) - want_phi)
        knot_p, knot_phi = desired(grid)
        knot_position_error = np.linalg.norm(np.array([arm.fk(v) for v in q]) - knot_p, axis=1)
        knot_angle_error = np.abs(pitch_of(q) - knot_phi)
        # Unreachable input poses retain their residual; refinement only limits
        # the additional error introduced between those poses.
        extra_position = np.maximum(0, position_error - np.interp(probes, grid, knot_position_error))
        extra_angle = np.maximum(0, angle_error - np.interp(probes, grid, knot_angle_error))
        bad = ((extra_position > .0005) | (extra_angle > np.radians(.25))).reshape(-1, 3).any(1)
        if not bad.any() or iteration == 6:
            break
        midpoints = ((grid[:-1] + grid[1:]) / 2)[bad]
        grid = np.sort(np.r_[grid, midpoints])
        want_p, want_phi = desired(grid)
        q = fit_joint_path(arm, want_p, want_phi, reference_q=curve(grid), locked=locked, w_pitch=w_pitch)

    plan = retime_joint_path(q, grid, lower=arm.lo, upper=arm.hi, v_pos=arm.v_pos, v_neg=arm.v_neg,
                            acceleration=np.maximum(arm.v_pos, arm.v_neg) / ramp_s, dt=dt)
    plan.metrics['cartesian_refinement'] = {
        'input_waypoints': original_count, 'refined_waypoints': len(grid), 'iterations': iteration,
        'converged': bool(not bad.any()),
        'max_sampled_extra_position_error_m': float(extra_position.max()),
        'max_sampled_extra_pitch_error_deg': float(np.degrees(extra_angle.max())),
        'note': 'Quarter/midpoint checks against the input Cartesian path; not a continuous error bound.',
    }
    return plan
