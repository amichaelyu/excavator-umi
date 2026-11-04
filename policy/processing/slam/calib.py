"""Camera/IMU time offset, rotation and gyro bias from the gyro vs. the camera rotation seen in an AprilGrid take.

AprilGrid poses on every frame of a calibration take (:func:`rotation_intervals`) are compared, over camera pairs
~100 ms apart, against the gyro integrated over exactly the same (shifted) endpoints; one take, fit on its first part
and checked on the rest (:func:`fit_offset`, with ambiguity and sensitivity checks).

python -m policy.processing.camera_calibration runs it and writes the result to configs/hardware/camera-calibration/camera.json ``camera_imu``.

Conventions:
  matching_imu_host_time = raw_camera_time + offset_s
  omega_camera = R_cam_imu @ (omega_gyro - bias);  R_imu_cam = R_cam_imu.T
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


class CalibrationError(ValueError):
    pass


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
