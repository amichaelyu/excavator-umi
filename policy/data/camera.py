"""The recording camera's calibration: configs/hardware/camera-calibration/camera.json, the bucket fisheye camera (1280x720) and its IMU.

One file that ``python -m policy.processing.camera_calibration`` writes and every reader takes the camera from:

    target      the AprilGrid the lens was fitted on: layout (Kalibr aprilgrid keys), measured grid_mm, how measured
    intrinsics  OpenCV fisheye (pinhole-equidistant): matrix, distortion k1..k4, image_size, fit statistics, source
    camera_imu  camera-to-gyro offset_s, R_cam_imu and gyro_bias_dps (sign_convention), check statistics, source

Readers: policy.processing.slam (Fisheye.load; the default camera/IMU calibration of the SLAM pipeline and
reconstruct_orbslam), digital_twin.sim.self_mask (the lens of the rendered mask). ``$UMI_EXCAVATOR_CAMERA`` names another file (like ``$UMI_EXCAVATOR_HARDWARE_MAP``).

Standard library and numpy only, so the hub, the sim and the processing image import it alike.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from policy.data.local_config import check_writable, ensure_local

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA = REPO / 'configs' / 'hardware' / 'camera-calibration' / 'camera.json'
CAMERA_ENV = 'UMI_EXCAVATOR_CAMERA'
SECTIONS = ('target', 'intrinsics', 'camera_imu')
SIGN = 'matching_imu_host_time = raw_camera_time + offset_s; omega_cam = R_cam_imu (omega_gyro - bias)'
FRAME_SHAPE = (720, 1280)   # (h, w) of a bucket camera frame; intrinsics.image_size is the same as [w, h]


def camera_path() -> Path:
    """The camera calibration in use: ``$UMI_EXCAVATOR_CAMERA``, else configs/hardware/camera-calibration/camera.json (per machine; created
    from configs/hardware/camera-calibration/camera.example.json when missing)."""
    path = os.environ.get(CAMERA_ENV)
    return Path(path) if path else ensure_local(DEFAULT_CAMERA)


def load(path=None) -> dict:
    return validate(json.loads(Path(path or camera_path()).read_text()))


def intrinsics(path=None) -> dict:
    """camera_model, distortion_model, matrix, distortion, image_size (and the fit's provenance)."""
    return load(path)['intrinsics']


def camera_imu(path=None) -> dict:
    """offset_s, R_cam_imu, gyro_bias_dps (and the fit's provenance); conventions in ``sign_convention``."""
    return load(path)['camera_imu']


def _finite(values, n=None) -> bool:
    values = list(values)
    return (n is None or len(values) == n) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def validate(cfg: dict) -> dict:
    """Light checks of the sections present; raises ValueError."""
    t = cfg.get('target')
    if t is not None and not (t.get('target_type') == 'aprilgrid' and int(t['tagCols']) > 0 and int(t['tagRows']) > 0
                              and float(t['tagSize']) > 0 and float(t['tagSpacing']) > 0):
        raise ValueError('camera target must be an aprilgrid with positive tagCols, tagRows, tagSize, tagSpacing')
    c = cfg.get('intrinsics')
    if c is not None:
        K = np.asarray(c['matrix'], float)
        size = c['image_size']
        if c.get('camera_model') != 'pinhole' or c.get('distortion_model') not in ('equidistant', 'radtan'):
            raise ValueError('camera intrinsics must be pinhole with equidistant (or radtan) distortion')
        n = 4 if c['distortion_model'] == 'equidistant' else 5
        if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
                or not _finite(c['distortion'], n) or len(size) != 2 or not all(type(v) is int and v > 0 for v in size)):
            raise ValueError(f'camera intrinsics need a 3x3 matrix, {n} distortion coefficients and an image size')
    m = cfg.get('camera_imu')
    if m is not None:
        R = np.asarray(m['R_cam_imu'], float)
        if not (_finite([m['offset_s']]) and abs(m['offset_s']) < 0.3 and _finite(m['gyro_bias_dps'], 3)
                and R.shape == (3, 3) and np.allclose(R @ R.T, np.eye(3), atol=1e-3) and np.linalg.det(R) > 0):
            raise ValueError('camera_imu needs offset_s (|s| < 0.3), a rotation R_cam_imu and 3 gyro_bias_dps')
    return cfg


def dumps(cfg: dict) -> str:
    """JSON, two-space indent, lists of plain values on one line."""
    text = json.dumps(cfg, indent=2, allow_nan=False)
    return re.sub(r'\[[^\[\]{}"]*\]', lambda m: '[' + ', '.join(filter(None, (x.strip() for x in m.group()[1:-1].split(',')))) + ']',
                  text) + '\n'


def update(sections: dict, path=None) -> dict:
    """Replace whole sections (``target``, ``intrinsics``, ``camera_imu``) and keep the rest of the file: re-read just
    before writing (another tool may have changed it since), validate, replace atomically. Returns the new config."""
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f'unknown camera sections {sorted(unknown)} (allowed: {list(SECTIONS)})')
    p = check_writable(path or camera_path())
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg.update(sections)
    validate(cfg)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f'.{p.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(dumps(cfg))
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return cfg
