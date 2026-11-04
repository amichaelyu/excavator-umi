"""Camera calibration: lens and camera/IMU of the bucket fisheye camera, straight into configs/hardware/camera-calibration/camera.json.

    uv run python -m policy.processing.camera_calibration <recording.tar | video> [options]

From a take of the AprilGrid (calibration-target/aprilgrid.svg; layout and measured size in configs/hardware/camera-calibration/camera.json
``target``): any file OpenCV can read (.mp4, .mov, .mjpg, ...) or a handheld-recorder archive (recording.tar with
metadata.json, frames.csv, imu.csv and video.mjpg). Frames are sampled, the grid is detected with OpenCV's AprilTag
36h11 dictionary, blurry and near-duplicate views are dropped and OpenCV's fisheye model (pinhole-equidistant,
k1..k4) is fitted: ``intrinsics``. A recorder archive also gives ``camera_imu``: target poses on every frame against
the gyro (policy.processing.slam.calib.fit_offset), fitted on the first 60% of the moving segment and checked on the
rest; written only when that check passes.

configs/hardware/camera-calibration/camera.json is updated in place (``--dry-run``: not at all), each section with its source recording, and
the raw outputs go to ``--output`` (default configs/hardware/camera-calibration/<recording id>/ (gitignored)):

    camchain.yaml   camchain (cam0: pinhole / equidistant), for Kalibr / ORB-SLAM3 tools
    results.txt     parameters with 1-sigma uncertainty, reprojection error, view statistics
    views.csv       every sampled frame where the grid was found, with why it was used or dropped
    time-check.json the camera/IMU fit and its held-out check (recorder archives)
    camera.json     the sections this run wrote (or, with --dry-run, would have written)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from policy.data import camera

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / 'configs' / 'hardware' / 'camera-calibration'
INTRINSICS_METHOD = 'OpenCV fisheye (pinhole-equidistant) fit to AprilGrid detections'
TARGET_METHOD = ('target poses (pose RMS <= 1.5 px) vs. the gyro integrated over the same ~100 ms intervals; fit on the '
                 'first 60% of the moving segment, checked on the rest; gyro bias fitted with them, from the stillest '
                 '3 s window as the start')
# AprilGrid corner order per tag is bottom-left, bottom-right, top-right, top-left (target frame: x right, y up,
# origin at tag 0's bottom-left corner). OpenCV returns the corners of the grid's tags (drawn with rotation=2) as
# bottom-right, bottom-left, top-left, top-right, so OpenCV corner k is grid corner OPENCV_TO_GRID[k].
OPENCV_TO_GRID = (1, 0, 3, 2)
# AprilGrid detection checks (the same as Kalibr's GridCalibrationTargetAprilgrid).
MAX_SUBPIX_DISPLACEMENT2 = 1.5
MIN_BORDER_DISTANCE = 4.0


# ------------------------------------------------------------------------------------------------ target


@dataclass(frozen=True)
class Target:
    cols: int
    rows: int
    size: float      # tag edge, metres
    spacing: float   # gap / tag edge

    @property
    def pitch(self) -> float:
        return self.size * (1 + self.spacing)

    @property
    def extent(self) -> tuple[float, float]:
        """Outer black edge to outer black edge (width, height), metres."""
        return (self.size * (self.cols + (self.cols - 1) * self.spacing),
                self.size * (self.rows + (self.rows - 1) * self.spacing))

    @property
    def tags(self) -> int:
        return self.cols * self.rows


def load_target(target: dict | None = None, grid_mm: float | None = None) -> Target:
    """``target`` (default: configs/hardware/camera-calibration/camera.json's) as a Target; ``grid_mm`` (measured grid width) overrides tagSize."""
    raw = camera.load()['target'] if target is None else target
    if raw.get('target_type') != 'aprilgrid':
        raise ValueError('only aprilgrid targets are supported')
    t = Target(int(raw['tagCols']), int(raw['tagRows']), float(raw['tagSize']), float(raw['tagSpacing']))
    if grid_mm is not None:
        if t.cols != t.rows:
            raise ValueError('--grid-mm assumes a square grid')
        t = Target(t.cols, t.rows, grid_mm / 1000 / (t.cols + (t.cols - 1) * t.spacing), t.spacing)
    return t


def target_keys(t: Target) -> dict:
    return {'target_type': 'aprilgrid', 'tagCols': t.cols, 'tagRows': t.rows, 'tagSize': t.size, 'tagSpacing': t.spacing}


def object_points(t: Target) -> np.ndarray:
    """All 4 * tags corner positions (N, 3), indexed by grid point id: row r of 2*rows, column c of 2*cols,
    id = r * 2*cols + c, at x = (c//2)*pitch + (c%2)*size, y likewise, z = 0."""
    r, c = np.mgrid[0:2 * t.rows, 0:2 * t.cols]
    x = (c // 2) * t.pitch + (c % 2) * t.size
    y = (r // 2) * t.pitch + (r % 2) * t.size
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype(np.float64)


def corner_ids(t: Target, tag: int) -> np.ndarray:
    """Grid point ids of a tag's corners: bottom-left, bottom-right, top-right, top-left."""
    width = 2 * t.cols
    base = (tag // t.cols) * 2 * width + (tag % t.cols) * 2
    return np.array([base, base + 1, base + width + 1, base + width])


# ------------------------------------------------------------------------------------------------ detection


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2   # the grid's 36h11 tags have a two-bit black border
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE   # refined below
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)


def detect(gray: np.ndarray, t: Target, detector, min_tags: int = 6):
    """Grid corners in one image: (grid point ids (N,), pixel positions (N, 2)), or None."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    ids = ids.ravel()
    counts = np.bincount(ids, minlength=t.tags)
    h, w = gray.shape
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    point_ids, points = [], []
    for tag, quad in zip(ids, corners):
        if tag >= t.tags or counts[tag] != 1:   # not part of this grid, or seen twice
            continue
        raw = quad.reshape(4, 2).astype(np.float32)
        if np.any(raw < MIN_BORDER_DISTANCE) or np.any(raw[:, 0] > w - MIN_BORDER_DISTANCE) \
                or np.any(raw[:, 1] > h - MIN_BORDER_DISTANCE):
            continue
        refined = cv2.cornerSubPix(gray, raw.copy().reshape(-1, 1, 2), (2, 2), (-1, -1), criteria).reshape(4, 2)
        if np.any(np.sum((refined - raw) ** 2, axis=1) > MAX_SUBPIX_DISPLACEMENT2):
            continue
        point_ids.append(corner_ids(t, int(tag))[list(OPENCV_TO_GRID)])
        points.append(refined)
    if len(point_ids) < min_tags:
        return None
    order = np.argsort(np.concatenate(point_ids))
    return np.concatenate(point_ids)[order], np.concatenate(points).astype(np.float64)[order]


def sharpness(gray: np.ndarray, points: np.ndarray) -> float:
    """Variance of the Laplacian over the grid's bounding box."""
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 1
    crop = gray[max(y0, 0):y1, max(x0, 0):x1]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


# ------------------------------------------------------------------------------------------------ input


def read_recording(path: Path):
    with tarfile.open(path, 'r:') as archive:
        meta = json.load(archive.extractfile('metadata.json'))
        frames = list(csv.DictReader(io.TextIOWrapper(archive.extractfile('frames.csv'))))
        imu = list(csv.DictReader(io.TextIOWrapper(archive.extractfile('imu.csv'))))
    return meta, frames, imu


def frames(path: Path, sample_hz: float, decode_all: bool = False):
    """Yield (frame index, time s, sampled for intrinsics, grayscale image), sampling sample_hz frames per second.
    Unsampled frames come with a None image unless decode_all. Recorder times are raw camera timestamps relative
    to metadata started_monotonic_s."""
    period = 1 / sample_hz
    next_t = -np.inf

    def take(time):
        nonlocal next_t
        if time + 1e-6 >= next_t:
            next_t = time + period
            return True
        return False

    if tarfile.is_tarfile(path):
        meta, rows, _ = read_recording(path)
        origin = meta['started_monotonic_s']
        with tarfile.open(path, 'r:') as archive:
            video = archive.extractfile('video.mjpg')
            for index, row in enumerate(rows):
                time = int(row['camera_timestamp_ns']) / 1e9 - origin
                sampled = take(time)
                if not (sampled or decode_all):
                    yield index, time, False, None
                    continue
                video.seek(int(row['offset']))
                image = cv2.imdecode(np.frombuffer(video.read(int(row['size'])), np.uint8), cv2.IMREAD_GRAYSCALE)
                yield index, time, sampled, image
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    index = 0
    while cap.grab():
        time = index / fps
        sampled = take(time)
        if sampled or decode_all:
            ok, image = cap.retrieve()
            yield index, time, sampled, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if ok else None
        else:
            yield index, time, False, None
        index += 1


# ------------------------------------------------------------------------------------------------ view selection


def view_distance(a: dict, b: dict, far: float) -> float:
    """RMS pixel motion of the corners two views share (``far`` when they share fewer than 8)."""
    common, ia, ib = np.intersect1d(a['ids'], b['ids'], assume_unique=True, return_indices=True)
    if len(common) < 8:
        return far
    return float(np.sqrt(np.mean(np.sum((a['points'][ia] - b['points'][ib]) ** 2, axis=1))))


def select_views(views: list[dict], min_motion: float, max_views: int, far: float) -> list[int]:
    """Greedy farthest-point selection: start from the view with most corners, then repeatedly add the view
    farthest from every selected one. Stops at max_views or when the farthest is within min_motion pixels (a
    near-duplicate of a selected view)."""
    if not views:
        return []
    chosen = [max(range(len(views)), key=lambda i: (len(views[i]['ids']), views[i]['sharpness']))]
    nearest = np.array([view_distance(v, views[chosen[0]], far) for v in views])
    nearest[chosen[0]] = -1
    while len(chosen) < max_views:
        i = int(np.argmax(nearest))
        if nearest[i] < min_motion:
            break
        chosen.append(i)
        nearest = np.minimum(nearest, [view_distance(v, views[i], far) for v in views])
        nearest[chosen] = -1
    return sorted(chosen)


# ------------------------------------------------------------------------------------------------ calibration


def fisheye_calibrate(obj: list[np.ndarray], img: list[np.ndarray], size: tuple[int, int]):
    """cv2.fisheye.calibrate, dropping views it reports as ill-conditioned. Returns (rms, K, D, rvecs, tvecs, kept
    indices)."""
    keep = list(range(len(obj)))
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_CHECK_COND
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    while True:
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                # OpenCV 5 wants each view as a (1, N, 3) / (1, N, 2) array.
                [obj[i].reshape(1, -1, 3) for i in keep], [img[i].reshape(1, -1, 2) for i in keep], size,
                np.eye(3), np.zeros(4), flags=flags, criteria=criteria)
            return rms, K, D.ravel(), list(rvecs), list(tvecs), keep
        except cv2.error as error:
            bad = re.search(r'input array (\d+)', str(error))
            if not bad or len(keep) <= 10:
                if flags & cv2.CALIB_CHECK_COND:
                    flags &= ~cv2.CALIB_CHECK_COND
                    continue
                raise
            del keep[int(bad.group(1))]


def view_residuals(obj, img, rvec, tvec, K, D):
    projected, jacobian = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
    return projected.reshape(-1, 2) - img, jacobian


def intrinsic_std(obj, img, rvecs, tvecs, K, D) -> np.ndarray:
    """1-sigma of [fx, fy, cx, cy, k1..k4]: Gauss-Newton covariance with the per-view poses marginalised (Schur
    complement), scaled by the residual variance."""
    S = np.zeros((8, 8))
    sse = n = 0
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        e, J = view_residuals(o, i, r, t, K, D)
        Ji, Je = J[:, :8], J[:, 8:14]   # jacobian columns: f(2) c(2) k(4) rvec(3) tvec(3) skew(1)
        W = Ji.T @ Je
        S += Ji.T @ Ji - W @ np.linalg.solve(Je.T @ Je, W.T)
        sse += float(np.sum(e ** 2))
        n += e.size
    sigma2 = sse / max(n - 8 - 6 * len(obj), 1)
    return np.sqrt(np.diag(np.linalg.inv(S)) * sigma2)


def coverage(points: list[np.ndarray], size: tuple[int, int], cells=(16, 9)) -> float:
    """Fraction of a cells[0] x cells[1] grid over the image that contains at least one used corner."""
    hit = np.zeros(cells[::-1], bool)
    for p in points:
        cx = np.clip((p[:, 0] / size[0] * cells[0]).astype(int), 0, cells[0] - 1)
        cy = np.clip((p[:, 1] / size[1] * cells[1]).astype(int), 0, cells[1] - 1)
        hit[cy, cx] = True
    return float(hit.mean())


def calibrate(views: list[dict], t: Target, size: tuple[int, int], min_views: int, outlier_px: float):
    """Fit, then drop views whose RMS exceeds max(outlier_px, 3 x median) and refit (up to three rounds)."""
    points = object_points(t)
    active = list(range(len(views)))
    rejected = {}
    for round_ in range(4):
        obj = [points[views[i]['ids']] for i in active]
        img = [views[i]['points'] for i in active]
        rms, K, D, rvecs, tvecs, kept = fisheye_calibrate(obj, img, size)
        for k in sorted(set(range(len(active))) - set(kept)):
            rejected[active[k]] = 'ill-conditioned'
        active = [active[k] for k in kept]
        obj, img = [obj[k] for k in kept], [img[k] for k in kept]
        per_view = [float(np.sqrt(np.mean(np.sum(view_residuals(o, i, r, tv, K, D)[0] ** 2, axis=1))))
                    for o, i, r, tv in zip(obj, img, rvecs, tvecs)]
        limit = max(outlier_px, 3 * float(np.median(per_view)))
        bad = [k for k, e in enumerate(per_view) if e > limit]
        if not bad or round_ == 3 or len(active) - len(bad) < min_views:
            break
        for k in bad:
            rejected[active[k]] = f'outlier ({per_view[k]:.2f} px)'
        active = [a for k, a in enumerate(active) if k not in bad]
    if len(active) < min_views:
        raise SystemExit(f'Only {len(active)} usable views (need {min_views}). Record more varied views of the grid.')
    residuals = np.concatenate([view_residuals(o, i, r, tv, K, D)[0] for o, i, r, tv in zip(obj, img, rvecs, tvecs)])
    return {'K': K, 'D': D, 'rms': float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
            'opencv_rms': float(rms), 'std': intrinsic_std(obj, img, rvecs, tvecs, K, D),
            'residual_mean': residuals.mean(axis=0), 'residual_std': residuals.std(axis=0),
            'corners': len(residuals), 'used': active, 'per_view_rms': dict(zip(active, per_view)),
            'rejected': rejected, 'coverage': coverage(img, size)}


# ------------------------------------------------------------------------------------------------ outputs


def intrinsics(K) -> list[float]:
    return [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]


def camchain(K, D, size) -> dict:
    return {'cam0': {'cam_overlaps': [], 'camera_model': 'pinhole', 'distortion_coeffs': [float(d) for d in D],
                     'distortion_model': 'equidistant', 'intrinsics': intrinsics(K), 'resolution': list(size),
                     'rostopic': '/cam0/image_raw'}}


def intrinsics_section(fit, size, provenance: dict) -> dict:
    """configs/hardware/camera-calibration/camera.json ``intrinsics`` from a fit."""
    fx, fy, cx, cy = intrinsics(fit['K'])
    std = [float(x) for x in fit['std']]
    return {'camera_model': 'pinhole', 'distortion_model': 'equidistant', 'matrix': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            'distortion': [float(d) for d in fit['D']], 'image_size': list(size),
            'std': {'fx_fy_cx_cy': std[:4], 'distortion': std[4:]}, 'reprojection_rms_px': fit['rms'],
            'corners': fit['corners'], 'views': len(fit['used']), 'coverage': round(fit['coverage'], 3),
            'method': f'{INTRINSICS_METHOD}, {len(fit["used"])} selected views', **provenance}


def results_text(fit, t: Target, source: str, counts: dict) -> str:
    fmt = lambda v: '[' + ' '.join(f'{x:.8g}' for x in v) + ']'   # noqa: E731
    std = fit['std']
    return f"""Calibration results
====================
Camera-system parameters:
cam0 ({source}):
    type: OpenCV fisheye (pinhole-equidistant)
    distortion: {fmt(fit['D'])} +- {fmt(std[4:])}
    projection: {fmt(intrinsics(fit['K']))} +- {fmt(std[:4])}
    reprojection error: {fmt(fit['residual_mean'])} +- {fmt(fit['residual_std'])}
    reprojection RMS: {fit['rms']:.4f} px over {fit['corners']} corners in {len(fit['used'])} views

Target configuration
====================

  Type: aprilgrid
  Tags:
    Rows: {t.rows}
    Cols: {t.cols}
    Size: {t.size} [m]
    Spacing {t.size * t.spacing} [m]
    Grid: {t.extent[0] * 1000:.2f} x {t.extent[1] * 1000:.2f} mm (outer black edge)

Views
=====
  frames: {counts['frames']}, sampled: {counts['sampled']}, grid detected: {counts['detected']}
  dropped blurry: {counts['blurry']}, near-duplicate or over --max-views: {counts['duplicate']}
  calibrated on: {len(fit['used'])}, rejected by the fit: {len(fit['rejected'])}
  image coverage: {fit['coverage'] * 100:.0f}% of 16 x 9 cells hold a used corner
"""


# ------------------------------------------------------------------------------------------------ camera / IMU


LIMITATIONS = ['Camera-to-gyro timing and rotation only, not camera-to-IMU translation.',
               'Host I2C read timestamps, rolling shutter and auto exposure remain sources of error.']


def time_offset(path: Path, detections: list[dict], t: Target, K, D) -> dict:
    """Camera-to-gyro time offset, rotation and gyro bias of a recorder take (time-check.json): target poses (fitted
    intrinsics, pose RMS <= 1.5 px) on every frame, rotations ~100 ms apart against the integrated gyro, fit on the
    first 60% of the moving segment and checked on the rest."""
    from scipy.spatial.transform import Rotation
    from policy.processing.slam.calib import CalibrationError, GyroAverager, fit_offset, rotation_intervals

    meta, _, imu = read_recording(path)
    origin = meta['started_monotonic_s']
    points = object_points(t)
    poses = []
    for d in detections:
        obj = points[d['ids']]
        normalized = cv2.fisheye.undistortPoints(d['points'].reshape(-1, 1, 2), K, D)
        ok, r, tv = cv2.solvePnP(obj, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tv[2, 0] <= 0:
            continue
        error = float(np.sqrt(np.mean(np.sum(view_residuals(obj, d['points'], r, tv, K, D)[0] ** 2, axis=1))))
        if error <= 1.5:
            poses.append((d['time'], cv2.Rodrigues(r)[0], error))   # target-to-camera rotation
    poses.sort(key=lambda p: p[0])
    it = np.array([int(r['read_monotonic_ns']) / 1e9 - origin for r in imu])
    gyro = np.deg2rad([[float(r[k]) for k in ('gx_dps', 'gy_dps', 'gz_dps')] for r in imu])
    end = min(poses[-1][0] if poses else 0, it[-1])
    windows = [(a, gyro[(it >= a) & (it < a + 3)]) for a in (0, 1, 2, end - 5, end - 4, end - 3)]
    windows = [(a, g) for a, g in windows if len(g) >= 100]
    result = {'recording_id': meta.get('id'), 'pose_frames': len(poses), 'detected_frames': len(detections),
              'sign_convention': camera.SIGN}
    try:
        if not windows:
            raise CalibrationError('Missing stationary gyro samples. Hold still at the beginning and end.')
        still_start, still = min(windows, key=lambda w: np.max(np.std(w[1], axis=0)))
        bias = np.median(still, axis=0)
        split = 5 + (end - 10) * .6
        s, e, v = rotation_intervals(poses, 5, end - 5)
        if len(s) < 200:
            raise CalibrationError('Too little continuous target tracking. Keep the pattern visible through the turns.')
        train, valid = e < split, s >= split
        fit = fit_offset(s[train], e[train], v[train], it, gyro, bias)
        check = fit_offset(s[valid], e[valid], v[valid], it, gyro, bias)
        p = fit['params']
        predicted = (GyroAverager(it, gyro)(s[valid], e[valid], p[0]) - p[4:]) @ Rotation.from_rotvec(p[1:4]).as_matrix().T
        observed = v[valid]
        rms = float(np.sqrt(np.mean(np.sum((predicted - observed) ** 2, axis=1))))
        motion = float(np.sqrt(np.mean(np.sum(observed ** 2, axis=1))))
        corr = float(np.corrcoef(np.linalg.norm(predicted, axis=1), np.linalg.norm(observed, axis=1))[0, 1])
        disagreement = abs(p[0] - check['params'][0]) * 1000
        sensitivity = min(fit['sensitivity'], check['sensitivity'])
        reasons = []
        if disagreement > 10: reasons.append('Training and held-out time offsets differ by more than 10 ms.')
        if rms > np.deg2rad(10) or rms / max(motion, 1e-6) > .25: reasons.append('Held-out angular motion residual is too high.')
        if not np.isfinite(corr) or corr < .9: reasons.append('Held-out rotation-speed correlation is below 0.90.')
        if sensitivity < 1.05: reasons.append('The timing estimate is weakly constrained. Vary rotation speed more.')
        if fit['ambiguous'] or check['ambiguous']: reasons.append('More than one plausible time offset fits this motion.')
        result.update(status='passed_internal_timing_check' if not reasons else 'rejected', reasons=reasons,
                      offset_s=float(p[0]), held_out_offset_ms=float(check['params'][0] * 1000),
                      offset_disagreement_ms=float(disagreement), held_out_rotation_rms_dps=float(np.rad2deg(rms)),
                      rotation_speed_correlation=corr, timing_sensitivity_ratio=float(sensitivity),
                      training_pairs=int(train.sum()), held_out_pairs=int(valid.sum()),
                      bias_stationary_window_s=[float(still_start), float(still_start + 3)],
                      gyro_bias_dps=np.rad2deg(p[4:]).tolist(), R_cam_imu=Rotation.from_rotvec(p[1:4]).as_matrix().tolist())
    except CalibrationError as error:
        result.update(status='rejected', reasons=[str(error)])
    result['limitations'] = [*LIMITATIONS, 'The holdout is within one take, and the intrinsics are fitted on the same take.']
    return result


CHECK_KEYS = ('status', 'held_out_offset_ms', 'offset_disagreement_ms', 'held_out_rotation_rms_dps',
              'rotation_speed_correlation', 'timing_sensitivity_ratio', 'training_pairs', 'held_out_pairs')


def camera_imu_section(check: dict, provenance: dict) -> dict:
    """configs/hardware/camera-calibration/camera.json ``camera_imu`` from a passed time_offset() check."""
    return {'offset_s': check['offset_s'], 'R_cam_imu': check['R_cam_imu'], 'gyro_bias_dps': check['gyro_bias_dps'],
            'sign_convention': camera.SIGN, 'translation': 'not estimated; assumed 0 (camera and IMU on one rigid mount)',
            'method': TARGET_METHOD, 'check': {k: check[k] for k in CHECK_KEYS}, **provenance,
            'limitations': check['limitations']}


# ------------------------------------------------------------------------------------------------ main


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def shown(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def write_camera(sections: dict, output: Path, config: Path, dry_run: bool):
    (output / 'camera.json').write_text(camera.dumps(sections))
    if dry_run:
        print(f'--dry-run: {shown(config)} not changed; it would get {", ".join(sections)} ({shown(output / "camera.json")})')
    else:
        camera.update(sections, config)
        print(f'Updated {", ".join(sections)} in {shown(config)}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='python -m policy.processing.camera_calibration', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path, help='video file or recorder archive (.tar)')
    p.add_argument('--output', type=Path, help='directory for the raw outputs (default configs/hardware/camera-calibration/<id>/)')
    p.add_argument('--config', type=Path, default=camera.camera_path(), help='camera calibration to update (default %(default)s)')
    p.add_argument('--dry-run', action='store_true', help='write the raw outputs only; leave --config unchanged')
    p.add_argument('--grid-mm', type=float, help='measured outer width of the black grid; overrides tagSize (and is written to target)')
    p.add_argument('--no-imu', action='store_true', help='recorder archives: fit the lens only, not camera_imu')
    p.add_argument('--sample-hz', type=float, default=4., help='frames sampled per second of video (default 4)')
    p.add_argument('--min-tags', type=int, default=6, help='tags a view needs (default 6)')
    p.add_argument('--blur', type=float, default=0.4,
                   help='drop views whose sharpness is below this fraction of the median (default 0.4)')
    p.add_argument('--min-motion', type=float, default=12.,
                   help='RMS corner motion in px below which a view duplicates a selected one (default 12)')
    p.add_argument('--max-views', type=int, default=60, help='views passed to the fit (default 60)')
    p.add_argument('--min-views', type=int, default=15, help='fewest views accepted (default 15)')
    p.add_argument('--outlier-px', type=float, default=1.0,
                   help='views above max(this, 3 x median) RMS are refitted without (default 1.0)')
    a = p.parse_args(argv)
    video = a.source

    is_recording = tarfile.is_tarfile(video)
    with_imu = is_recording and not a.no_imu
    cfg = camera.load(a.config)
    t = load_target(cfg['target'], a.grid_mm)
    detector = make_detector()
    counts = dict(frames=0, sampled=0, detected=0, blurry=0, duplicate=0)
    sampled, every, size = [], [], None
    for index, time_s, is_sample, gray in frames(video, a.sample_hz, decode_all=with_imu):
        counts['frames'] += 1
        counts['sampled'] += is_sample
        if index % 200 == 0:
            print(f'frame {index}: {len(sampled)} sampled views with the grid so far', flush=True)
        if gray is None:
            continue
        frame_size = (gray.shape[1], gray.shape[0])
        if size not in (None, frame_size):
            raise SystemExit('The image size changed during the video.')
        size = frame_size
        found = detect(gray, t, detector, a.min_tags)
        if found is None:
            continue
        view = {'index': index, 'time': time_s, 'ids': found[0], 'points': found[1]}
        every.append(view)
        if is_sample:
            view['sharpness'] = sharpness(gray, found[1])
            sampled.append(view)
    counts['detected'] = len(sampled)
    if not sampled:
        raise SystemExit('The grid was not found in any sampled frame.')

    median = float(np.median([v['sharpness'] for v in sampled]))
    for v in sampled:
        v['status'] = 'blurry' if v['sharpness'] < a.blur * median else 'candidate'
    candidates = [v for v in sampled if v['status'] == 'candidate']
    counts['blurry'] = len(sampled) - len(candidates)
    chosen = set(select_views(candidates, a.min_motion, a.max_views, far=float(np.hypot(*size))))
    for k, v in enumerate(candidates):
        if k not in chosen:
            v['status'] = 'duplicate'
    counts['duplicate'] = len(candidates) - len(chosen)
    views = [candidates[k] for k in sorted(chosen)]
    print(f'{len(sampled)} sampled views with the grid; {counts["blurry"]} blurry, {counts["duplicate"]} '
          f'near-duplicate; fitting {len(views)}', flush=True)

    fit = calibrate(views, t, size, a.min_views, a.outlier_px)
    for k, v in enumerate(views):
        v['status'] = 'used' if k in fit['per_view_rms'] else fit['rejected'].get(k, 'rejected')
        v['rms'] = fit['per_view_rms'].get(k)

    recording_id = read_recording(video)[0].get('id') if is_recording else None
    output = a.output or RESULTS / (recording_id or video.stem)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'recording_id': recording_id, 'source': video.name, 'source_sha256': sha256(video),
                  'results': shown(output), 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (output / 'camchain.yaml').write_text(yaml.safe_dump(camchain(fit['K'], fit['D'], size),
                                                         default_flow_style=None, sort_keys=False))
    report = results_text(fit, t, video.name, counts)
    (output / 'results.txt').write_text(report)
    with (output / 'views.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame_index', 'time_s', 'corners', 'sharpness', 'status', 'rms_px'])
        for v in sampled:
            writer.writerow([v['index'], f"{v['time']:.6f}", len(v['ids']), f"{v['sharpness']:.1f}", v['status'],
                             '' if v.get('rms') is None else f"{v['rms']:.4f}"])
    print(report)
    sections = {'intrinsics': intrinsics_section(fit, size, provenance)}
    if a.grid_mm is not None:
        sections['target'] = {**cfg['target'], **target_keys(t), 'grid_mm': [a.grid_mm, a.grid_mm],
                              'measured': f'--grid-mm {a.grid_mm} for {provenance["source"]} (tape, outer black edge)'}
    if with_imu:
        check = time_offset(video, every, t, fit['K'], fit['D'])
        (output / 'time-check.json').write_text(json.dumps(check, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: check.get(k) for k in ('status', 'reasons', 'offset_s', 'held_out_offset_ms',
                                                     'rotation_speed_correlation')}, indent=2))
        if check['status'] == 'passed_internal_timing_check':
            sections['camera_imu'] = camera_imu_section(check, provenance)
        else:
            print('camera_imu not written: the held-out check did not pass (time-check.json reasons).')
    write_camera(sections, output, a.config, a.dry_run)
    print(f'Wrote {shown(output)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
