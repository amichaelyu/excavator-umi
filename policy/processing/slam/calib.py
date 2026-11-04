"""Camera/IMU time offset, rotation and gyro bias from the gyro vs. visually estimated camera rotation.

Two sources of camera rotation, compared over camera pairs ~100 ms apart against the gyro integrated over
exactly the same (shifted) endpoints:

  target poses     AprilGrid poses on every frame of a calibration take (:func:`rotation_intervals`), one take,
                   fit on its first part and checked on the rest (:func:`fit_offset`, with ambiguity and
                   sensitivity checks)
  natural features fisheye KLT tracks -> essential matrix on normalized coordinates (:func:`visual_rotation_pairs`),
                   any number of recordings, one offset and rotation with a gyro bias per recording (:func:`fit`)

python -m policy.processing.camera_calibration runs them (the first on a target take, the second with --natural) and
writes the result to configs/camera.json ``camera_imu``.

Conventions:
  matching_imu_host_time = raw_camera_time + offset_s
  omega_camera = R_cam_imu @ (omega_gyro - bias);  R_imu_cam = R_cam_imu.T
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from .frontend import Tracks, correspondences, two_view
from .recording import Fisheye, Recording


class CalibrationError(ValueError):
    pass


def visual_rotation_pairs(rec: Recording, tr: Tracks, cam: Fisheye, span=2, min_corr=40, min_inliers=30, max_rate_dps=300):
    """Return (t_a, t_b, omega_cam[N,3] rad/s, weight) for processed frame pairs (k, k+span)."""
    ta, tb, om, wt = [], [], [], []
    times = rec.cam_t[tr.frame_idx]
    for k in range(len(tr.obs) - span):
        common, pa, pb = correspondences(tr, k, k + span)
        if len(common) < min_corr:
            continue
        res = two_view(cam, pa, pb)
        if res is None:
            continue
        R_ab, t_ab, inl, info = res
        if info['inliers'] < min_inliers:
            continue
        dt = times[k + span] - times[k]
        if dt > 0.13:
            continue
        w = Rot.from_matrix(R_ab).as_rotvec() / dt
        if np.linalg.norm(w) > np.deg2rad(max_rate_dps):
            continue  # gross E-decomposition failure (gyro never exceeds ~150 dps here)
        ta.append(times[k]); tb.append(times[k + span])
        om.append(w)
        wt.append(info['inliers'])
    return np.array(ta), np.array(tb), np.array(om).reshape(-1, 3), np.array(wt)


class GyroAverager:
    """Mean gyro rate over [a + dt, b + dt]; ``strict``: raise if that leaves the IMU recording."""
    def __init__(self, t, gyro, strict=False):
        self.t, self.strict = t, strict
        self.I = cumulative_trapezoid(gyro, t, axis=0, initial=0)

    def __call__(self, a, b, dt):
        a = a + dt; b = b + dt
        if self.strict and (a.min() < self.t[0] or b.max() > self.t[-1]):
            raise CalibrationError('The IMU recording does not cover the camera timing search.')
        return np.column_stack([(np.interp(b, self.t, self.I[:, i]) - np.interp(a, self.t, self.I[:, i])) / (b - a)
                                for i in range(3)])


def kabsch(x, y):
    """R minimizing |R x - y|."""
    u, _, vt = np.linalg.svd(y.T @ x)
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def rotation_intervals(poses, start, end, span=.1):
    """(t_a, t_b, omega_cam[N,3] rad/s) from target-to-camera poses [(time, R, rms)] about ``span`` apart.

    Differencing adjacent 20 Hz poses amplifies independent corner/pose noise; the gyro is integrated over these
    exact endpoints, so this adds no moving-average delay to either stream. Never bridges a tracking gap > 120 ms.
    """
    times = np.array([p[0] for p in poses])
    starts, ends, visual = [], [], []
    for i, (a, ra, _) in enumerate(poses):
        j = np.searchsorted(times, a + span - .015)
        if j >= len(poses) or j <= i:
            continue
        b, rb, _ = poses[j]
        if a < start or b > end or not span - .02 < b - a < span + .03:
            continue
        if np.max(np.diff(times[i:j + 1])) > .12:
            continue
        # Target-to-camera poses rotate opposite the camera's body rotation.
        velocity = Rot.from_matrix(ra @ rb.T).as_rotvec() / (b - a)
        if np.linalg.norm(velocity) < np.deg2rad(250):
            starts.append(a); ends.append(b); visual.append(velocity)
    return np.array(starts), np.array(ends), np.array(visual).reshape(-1, 3)


def fit_offset(starts, ends, visual, imu_times, gyro, initial_bias=None):
    """One take: time offset, rotation and gyro bias (params [dt, rotvec(R_cam_imu), bias rad/s]) from four seeds of
    a speed-only offset search. Raises CalibrationError when the motion cannot constrain them; ``sensitivity`` (cost
    20 ms off / cost at the fit) and ``ambiguous`` (another seed fits almost as well) say how well it is pinned."""
    if len(starts) < 60:
        raise CalibrationError('Too few usable camera rotations. Keep the whole pattern visible and move for longer.')
    bias = np.zeros(3) if initial_bias is None else np.asarray(initial_bias)
    average = GyroAverager(imu_times, gyro, strict=True)
    candidates = np.linspace(-.3, .3, 301)
    speed = np.linalg.norm(visual, axis=1)
    if np.std(speed) < np.deg2rad(3):
        raise CalibrationError('Rotation speed barely changed. Vary your speed and pause between turns.')
    scores = np.array([np.mean((np.linalg.norm(average(starts, ends, dt) - bias, axis=1) - speed) ** 2) for dt in candidates])
    seeds = []
    for index in np.argsort(scores):
        dt = candidates[index]
        if all(abs(dt - existing) > .04 for existing in seeds):
            seeds.append(dt)
        if len(seeds) == 4:
            break

    def residual(p):
        return ((average(starts, ends, p[0]) - p[4:7]) @ Rot.from_rotvec(p[1:4]).as_matrix().T - visual).ravel()

    results = []
    for dt in seeds:
        p = np.r_[dt, Rot.from_matrix(kabsch(average(starts, ends, dt) - bias, visual)).as_rotvec(), bias]
        results.append(least_squares(residual, p, bounds=(np.r_[-.3, [-np.inf] * 3, bias - .12], np.r_[.3, [np.inf] * 3, bias + .12]),
                                     loss='soft_l1', f_scale=.035, x_scale='jac', max_nfev=150))
    best = min(results, key=lambda r: r.cost)
    p = best.x
    if not best.success or abs(p[0]) > .285:
        raise CalibrationError('The timing fit did not converge within its search range.')
    predicted = (average(starts, ends, p[0]) - p[4:7]) @ Rot.from_rotvec(p[1:4]).as_matrix().T
    error = predicted - visual
    singular = np.linalg.svd(visual - visual.mean(axis=0), compute_uv=False)
    if singular[-1] < .08 * singular[0]:
        raise CalibrationError('Motion did not cover all three rotation axes. Include left/right, up/down, and roll.')
    # A flat timing objective cannot justify a precise-looking offset.
    nearby = [np.mean(residual(np.r_[p[0] + shift, p[1:]]) ** 2) for shift in (-.02, .02)]
    sensitivity = min(nearby) / max(np.mean(error ** 2), 1e-12)
    ambiguous = any(r.cost < best.cost * 1.1 for r in results if abs(r.x[0] - p[0]) > .04)
    return {'params': p, 'rms': np.sqrt(np.mean(np.sum(error ** 2, axis=1))), 'sensitivity': sensitivity,
            'ambiguous': ambiguous, 'predicted': predicted}


def fit(datasets, search=(-0.15, 0.15), bias_prior=None, f_scale=0.05):
    """Jointly fit one time offset and one rotation, with one gyro bias per dataset.

    datasets: list of dicts with keys ta, tb, omega, avg (GyroAverager).
    Returns dict with offset_s, R_cam_imu, biases, residual stats and an offset profile.
    """
    nd = len(datasets)
    grid = np.arange(search[0], search[1] + 1e-9, 0.002)

    def speed_cost(dt):
        c = 0; n = 0
        for d in datasets:
            g = d['avg'](d['ta'], d['tb'], dt)
            g = g - np.median(g, axis=0) * 0  # bias unknown here; speed profile is dominated by motion
            c += np.sum((np.linalg.norm(g, axis=1) - np.linalg.norm(d['omega'], axis=1)) ** 2); n += len(g)
        return c / n
    prof_speed = np.array([speed_cost(dt) for dt in grid])
    dt0 = grid[np.argmin(prof_speed)]

    def residual(p, dt_fixed=None):
        dt = p[0] if dt_fixed is None else dt_fixed
        R = Rot.from_rotvec(p[1:4]).as_matrix()
        out = []
        for i, d in enumerate(datasets):
            b = p[4 + 3 * i: 7 + 3 * i]
            out.append(((d['avg'](d['ta'], d['tb'], dt) - b) @ R.T - d['omega']).ravel())
        return np.concatenate(out)

    # Jacobian sparsity: rows of dataset i depend on (offset, rotation) and bias_i only.
    from scipy.sparse import lil_matrix
    nrows = sum(3 * len(d['ta']) for d in datasets)
    S = lil_matrix((nrows, 4 + 3 * nd), dtype=int)
    r0 = 0
    for i, d in enumerate(datasets):
        n = 3 * len(d['ta'])
        S[r0:r0 + n, :4] = 1; S[r0:r0 + n, 4 + 3 * i:7 + 3 * i] = 1
        r0 += n
    S = S.tocsr()
    g_all = np.vstack([d['avg'](d['ta'], d['tb'], dt0) for d in datasets])
    v_all = np.vstack([d['omega'] for d in datasets])
    R0 = kabsch(g_all, v_all)
    p0 = np.r_[dt0, Rot.from_matrix(R0).as_rotvec(), np.zeros(3 * nd)]
    lo = np.r_[search[0], [-np.inf] * 3, [-0.2] * 3 * nd]; hi = np.r_[search[1], [np.inf] * 3, [0.2] * 3 * nd]
    best = None
    for seed in (dt0, dt0 - 0.03, dt0 + 0.03):
        p0[0] = np.clip(seed, search[0] + 1e-3, search[1] - 1e-3)
        r = least_squares(residual, p0, bounds=(lo, hi), loss='soft_l1', f_scale=f_scale, x_scale='jac',
                          **({'jac_sparsity': S, 'tr_solver': 'lsmr'} if nd > 1 else {}))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    # Offset profile with rotation/bias re-optimized at each offset (coarse)
    prof = []
    for dt in np.arange(p[0] - 0.05, p[0] + 0.0501, 0.005):
        rr = least_squares(lambda q: residual(np.r_[dt, q]), p[1:], loss='soft_l1', f_scale=f_scale,
                           **({'jac_sparsity': S[:, 1:], 'tr_solver': 'lsmr'} if nd > 1 else {}))
        prof.append((float(dt), float(np.sqrt(np.mean(rr.fun ** 2)))))
    res = residual(p).reshape(-1, 3)
    norms = np.linalg.norm(res, axis=1)
    motion = np.linalg.norm(v_all, axis=1)
    inl = norms < 3 * np.median(norms) + 1e-9
    # Rough 1-sigma of the offset from the Gauss-Newton covariance (robust residual scale)
    J = best.jac.toarray() if hasattr(best.jac, 'toarray') else best.jac
    try:
        s2 = np.median(best.fun ** 2) / 0.455
        cov = np.linalg.pinv(J.T @ J) * s2
        sd = np.sqrt(np.diag(cov))
    except Exception:
        sd = np.full(len(p), np.nan)
    R = Rot.from_rotvec(p[1:4]).as_matrix()
    pred = np.vstack([(d['avg'](d['ta'], d['tb'], p[0]) - p[4 + 3 * i:7 + 3 * i]) @ R.T for i, d in enumerate(datasets)])
    corr = float(np.corrcoef(np.linalg.norm(pred, axis=1), motion)[0, 1])
    return {
        'offset_s': float(p[0]), 'offset_sd_s': float(sd[0]),
        'R_cam_imu': R.tolist(), 'R_imu_cam': R.T.tolist(),
        'rotation_sd_deg': np.degrees(sd[1:4]).tolist(),
        'gyro_bias_dps': [np.degrees(p[4 + 3 * i:7 + 3 * i]).tolist() for i in range(nd)],
        'pairs': int(len(res)),
        'residual_rms_dps': float(np.degrees(np.sqrt(np.mean(norms ** 2)))),
        'residual_median_dps': float(np.degrees(np.median(norms))),
        'residual_rms_inliers_dps': float(np.degrees(np.sqrt(np.mean(norms[inl] ** 2)))),
        'motion_rms_dps': float(np.degrees(np.sqrt(np.mean(motion ** 2)))),
        'speed_correlation': corr,
        'offset_profile_rms_dps': [(dt, float(np.degrees(v))) for dt, v in prof],
        'speed_only_offset_s': float(dt0),
    }


def rotation_angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(np.asarray(Ra).T @ np.asarray(Rb)).as_rotvec())))
