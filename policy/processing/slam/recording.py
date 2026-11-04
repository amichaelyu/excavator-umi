"""Recording loader and fisheye camera model."""
from __future__ import annotations

import csv
import io
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from policy.data import camera
from policy.data.data_root import require_real_file, resolve_data_root

REPO = Path(__file__).resolve().parents[3]
DATASET = resolve_data_root()  # overridden by set_data_root() from the CLI --data-root flag
G = 9.80665

# Per-episode leading time to discard (seconds after the first camera frame).
# Episode 1 has a startup camera/IMU gap.
SKIP_START_S = {'20260927T070203Z-bf0ec1b6': 0.55}


@dataclass
class Fisheye:
    """Kannala-Brandt pinhole-equidistant model (OpenCV fisheye)."""
    K: np.ndarray
    D: np.ndarray
    size: tuple

    @classmethod
    def load(cls, path: Path | None = None) -> 'Fisheye':
        """The lens in configs/camera.json (``intrinsics``), or in another camera.json."""
        cam = camera.intrinsics(path)
        assert cam['distortion_model'] == 'equidistant'
        return cls(np.array(cam['matrix'], float), np.array(cam['distortion'], float), tuple(cam['image_size']))

    @property
    def fx(self): return self.K[0, 0]

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """Pixel points (N,2) -> normalized pinhole coordinates (N,2)."""
        if len(pts) == 0:
            return np.zeros((0, 2))
        out = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), self.K, self.D)
        return out.reshape(-1, 2)

    def bearings(self, pts: np.ndarray) -> np.ndarray:
        n = self.undistort(pts)
        b = np.column_stack([n, np.ones(len(n))])
        return b / np.linalg.norm(b, axis=1, keepdims=True)

    def project(self, P: np.ndarray) -> np.ndarray:
        """Camera-frame 3D points (N,3) -> pixels (N,2)."""
        P = np.asarray(P, float).reshape(-1, 3)
        r = np.hypot(P[:, 0], P[:, 1])
        th = np.arctan2(r, P[:, 2])
        k1, k2, k3, k4 = self.D
        thd = th * (1 + k1*th**2 + k2*th**4 + k3*th**6 + k4*th**8)
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        return np.column_stack([self.K[0, 0]*P[:, 0]*s + self.K[0, 2], self.K[1, 1]*P[:, 1]*s + self.K[1, 2]])


@dataclass
class Recording:
    id: str
    path: Path
    meta: dict
    cam_t: np.ndarray            # raw camera start-of-exposure time [s], monotonic clock
    cam_seq: np.ndarray
    offsets: np.ndarray
    sizes: np.ndarray
    imu_t: np.ndarray            # host I2C read midpoint [s]
    gyro: np.ndarray             # rad/s, sensor axes
    accel: np.ndarray            # m/s^2, sensor axes
    t0: float = 0.0              # episode origin (first kept camera frame, raw time)
    keep: np.ndarray = field(default=None)   # boolean mask of kept frames
    quality: dict = field(default_factory=dict)   # loader-detected data problems (see load())
    _video: bytes = field(default=None, repr=False)

    @property
    def n(self): return len(self.cam_t)

    def video(self) -> bytes:
        if self._video is None:
            with tarfile.open(self.path) as tar:
                self._video = tar.extractfile('video.mjpg').read()
        return self._video

    def decode(self, i: int, gray: bool = True):
        """Return (image or None, method). Tries OpenCV, then PIL with truncated-image tolerance."""
        v = self.video()
        a, n = int(self.offsets[i]), int(self.sizes[i])
        if a < 0 or n <= 0 or a >= len(v):
            return None, 'out_of_range'
        data = v[a:a + n]
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if img is not None and img.shape[:2] == camera.FRAME_SHAPE:
            return img, 'strict'
        try:
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im = Image.open(io.BytesIO(data))
            im.load()
            arr = np.asarray(im.convert('L' if gray else 'RGB'))
            if arr.shape[:2] != camera.FRAME_SHAPE:
                return None, 'bad_size'
            return (arr if gray else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)), 'lenient'
        except Exception:
            return None, 'failed'

    def image(self, i: int, gray: bool = True) -> np.ndarray:
        img, how = self.decode(i, gray)
        if img is None:
            raise ValueError(f'JPEG decode failed for frame {i} of {self.id} ({how}); load() should have dropped it')
        return img


def set_data_root(path=None) -> Path:
    """Re-resolve the dataset root (CLI flag > $UMI_DATA_DIR > the repository's dataset/)."""
    global DATASET
    DATASET = resolve_data_root(path)
    return DATASET


def episode_ids() -> list[str]:
    return sorted(p.name for p in DATASET.iterdir() if (p / 'recording.tar').exists())


def _rows(tar, name, fields):
    """Parse a CSV member tolerantly: rows that are truncated or unparseable are skipped and counted."""
    member = tar.extractfile(name)
    text = member.read().decode('utf-8', errors='replace')
    good, bad = [], 0
    for r in csv.DictReader(io.StringIO(text)):
        try:
            good.append([int(r[f]) if kind is int else float(r[f]) for f, kind in fields])
        except (TypeError, ValueError, KeyError):
            bad += 1
    return good, bad, not text.endswith('\n')


FRAME_FIELDS = [('camera_timestamp_ns', int), ('sequence', int), ('offset', int), ('size', int)]
IMU_FIELDS = [('read_monotonic_ns', int), ('gx_dps', float), ('gy_dps', float), ('gz_dps', float),
              ('ax_g', float), ('ay_g', float), ('az_g', float), ('clipped', int)]
IMU_GAP_S = 0.05          # a longer gap in the IMU stream ends the usable episode
GYRO_RANGE_DPS = 500.0    # LSM6DSL configured range


def load(episode: str, scan_images: bool = True) -> Recording:
    """Load a recording tolerantly.

    - Truncated / malformed CSV rows are skipped (dashboard restarts can cut the files).
    - IMU samples that are physically invalid (accel norm < 0.2 g, gyro beyond range, clipped) are removed;
      a disconnect typically leaves such a sample at the end.
    - The episode ends at the last valid IMU sample before any IMU gap > 50 ms (IMU disconnected).
    - Frames whose JPEG cannot be decoded even leniently are dropped (timestamps of the others unchanged, so
      the gap is preserved).
    Everything detected is recorded in rec.quality.
    """
    path = require_real_file(DATASET / episode / 'recording.tar')
    q = {}
    with tarfile.open(path) as tar:
        meta = json.load(tar.extractfile('metadata.json'))
        fr, bad_f, trunc_f = _rows(tar, 'frames.csv', FRAME_FIELDS)
        im, bad_i, trunc_i = _rows(tar, 'imu.csv', IMU_FIELDS)
    q['frames_csv_malformed_rows'] = bad_f; q['imu_csv_malformed_rows'] = bad_i
    q['frames_csv_truncated'] = trunc_f or bad_f > 0; q['imu_csv_truncated'] = trunc_i or bad_i > 0
    fr = np.array(fr, dtype=np.int64).reshape(-1, 4)
    im = np.array(im, dtype=float).reshape(-1, 8)
    # strictly increasing timestamps (drop repeats / reordering from truncated writes)
    fr = fr[np.r_[True, np.diff(fr[:, 0]) > 0]] if len(fr) else fr
    im = im[np.r_[True, np.diff(im[:, 0]) > 0]] if len(im) else im
    cam_t = fr[:, 0] * 1e-9
    imu_t = im[:, 0] * 1e-9
    gyro_dps = im[:, 1:4]; acc_g = im[:, 4:7]
    valid = (np.linalg.norm(acc_g, axis=1) > 0.2) & (np.max(np.abs(gyro_dps), axis=1) < GYRO_RANGE_DPS) & (im[:, 7] == 0)
    q['imu_invalid_samples_removed'] = int((~valid).sum())
    imu_t, gyro_dps, acc_g = imu_t[valid], gyro_dps[valid], acc_g[valid]
    rec = Recording(episode, path, meta, cam_t, fr[:, 1], fr[:, 2], fr[:, 3], imu_t,
                    np.deg2rad(gyro_dps), acc_g * G)
    start = max(cam_t[0] + SKIP_START_S.get(episode, 0.0), meta['started_monotonic_s'])
    # usable IMU span: the contiguous block (no gap > IMU_GAP_S) containing the start of the episode
    gaps = np.flatnonzero(np.diff(imu_t) > IMU_GAP_S)
    lo_i, hi_i = 0, len(imu_t) - 1
    for g in gaps:
        if imu_t[g + 1] <= start + 0.1:
            lo_i = g + 1
        elif hi_i == len(imu_t) - 1:
            hi_i = g
    imu_end = imu_t[hi_i]
    q['imu_gaps_over_50ms'] = int(len(gaps))
    q['imu_ended_early'] = bool(imu_end < cam_t[-1] - 0.1)
    if q['imu_ended_early']:
        q['truncated_at_last_imu_s_after_start'] = float(imu_end - start)
        q['camera_time_lost_s'] = float(cam_t[-1] - imu_end)
    # +-0.1 s margin for the time-offset search; after a startup IMU gap 30 ms (> offset) is enough.
    rec.keep = (cam_t >= start) & (cam_t > imu_t[lo_i] + (0.1 if lo_i == 0 else 0.03)) & (cam_t < imu_end - 0.1)
    # frames pointing outside video.mjpg (truncated video) or failing even lenient decoding
    undecodable, lenient = [], []
    if scan_images:
        for i in np.flatnonzero(rec.keep):
            img, how = rec.decode(int(i))
            if img is None:
                undecodable.append(int(i)); rec.keep[i] = False
            elif how == 'lenient':
                lenient.append(int(i))
    q['frames_undecodable'] = len(undecodable); q['undecodable_frame_indices'] = undecodable
    q['frames_lenient_decoded'] = len(lenient); q['lenient_frame_indices'] = lenient
    if undecodable:
        print(f'  {episode}: dropped {len(undecodable)} undecodable JPEG frame(s) {undecodable}', flush=True)
    if rec.keep.sum() < 20:
        raise ValueError(f'{episode}: only {int(rec.keep.sum())} usable frames after quality checks')
    rec.t0 = cam_t[rec.keep][0]
    rec.quality = q
    return rec


def manifest_quality(episode: str) -> dict:
    """Recorder/manifest flags for one episode (not used to skip anything)."""
    f = DATASET / 'manifest.json'
    if not f.exists():
        return {}
    for e in json.loads(f.read_text()).get('episodes', []):
        if e.get('id') == episode:
            return {k: e.get(k) for k in ('needs_review', 'validation_status', 'video_recovered', 'recorder_error',
                                          'recorder_warning', 'processing_note', 'label', 'task')}
    return {}
