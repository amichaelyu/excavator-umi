"""Replay a tracked (SLAM+IMU) bucket trajectory on the excavator digital twin.

Pipeline: load + clean + resample the camera trajectory to the 20 Hz control
rate -> apply the camera->bucket mount transform -> retarget the bucket path into
the model workspace with a similarity transform (yaw + uniform scale +
translation, world +z kept as model +z; the 3D path is kept, the slew swings the arm plane
to it) -> IK on ``[slew, boom, stick, bucket]`` for the ``tooth_tip`` position, with the
bucket pitch about the (slewed) arm axis -> path-wide braking and local slowdown
(``--planner lookahead``), or fixed-clock tracking (``--planner ik``) -> forward propagation,
both kinematically (qpos = action,
``mj_forward``) and dynamically (action fed to the position actuators, physics
stepped 10 x 5 ms per tick as in ``env.py``) -> CSV/JSON/PNG/MP4 outputs.
``--planner coupled`` instead plans button pulses for the real machine's coupled channels
(coupled.py).

Input CSV contract: ``t_s,px,py,pz,qw,qx,qy,qz,status``. Position is the camera
in a gravity-aligned world frame (+z up, metres); quaternion is world<-camera
(scalar first); status is one of tracked / imu_only / interpolated. The camera
(OpenCV axes: +z optical, +y down) is rigidly mounted on the bucket; the fixed
camera->bucket offset (pose of the bucket reference point in the camera frame) comes
from the ``camera_mount`` section of ``configs/hardware/robot.json`` (``--mount-file``) unless ``--cam-to-bucket`` is given.
While the file holds the unmeasured identity, the camera centre *is* the tooth-tip
target and a warning is printed.

Simulator-only. No robot commands are issued.

    python -m digital_twin.sim.replay --synthetic --output /tmp/replay-synthetic
    # what the labels stage (python -m policy labels <config>) runs per recording, with the config's options:
    python -m digital_twin.sim.replay --trajectory dataset/reconstructions/<config>/<id>/trajectory.csv \
        --output dataset/labels/<config>/<id>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .kinematics import BUCKET_ANGLE_WEIGHT, RAMP_S, SMOOTH_WEIGHT, CONTROL_HZ, FRAMESKIP, JOINTS, NQ, pitch_of, Arm, Envelope, PlanarChain, analytic_seed, locked_joints, propagate_dynamic, propagate_kinematic, rate_limit, reachable_envelope, solve_ik, track_ik
from ..paths import CONFIG
from .viz import add_segment, hud, make_camera, open_writer

COLUMNS = ['t_s', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'status']
STATUSES = ('tracked', 'imu_only', 'interpolated')


# --------------------------------------------------------------------------- I/O
def load_trajectory(path) -> tuple[pd.DataFrame, dict]:
    """Read a trajectory CSV; drop NaN / zero-quaternion / non-monotonic rows.

    Returns the cleaned frame (quaternions normalised, sign-continuous) and a
    report of what was dropped, with ``input_sha256``: the SHA-256 of the bytes parsed (policy.preprocessing
    checks labels against their trajectory.csv by it). A missing ``status`` column defaults to tracked.
    """
    data = Path(path).read_bytes()
    raw = pd.read_csv(io.BytesIO(data))
    raw.columns = [c.strip() for c in raw.columns]
    missing = [c for c in COLUMNS[:-1] if c not in raw.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    df = raw.copy()
    for c in COLUMNS[:-1]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'status' not in df.columns:
        df['status'] = 'tracked'
    df['status'] = df['status'].fillna('tracked').astype(str).str.strip().str.lower()
    report = {'input_sha256': hashlib.sha256(data).hexdigest(), 'rows_in': int(len(raw))}
    finite = np.isfinite(df[COLUMNS[:-1]].to_numpy(float)).all(axis=1)
    qn = np.linalg.norm(df[['qw', 'qx', 'qy', 'qz']].to_numpy(float), axis=1)
    ok = finite & (qn > 1e-6)
    report['dropped_nan_or_bad_quat'] = int((~ok).sum())
    df = df[ok].reset_index(drop=True)
    # Keep only strictly increasing timestamps (backwards jumps / duplicates are glitches).
    t = df['t_s'].to_numpy(float)
    prev_max = np.concatenate([[-np.inf], np.maximum.accumulate(t)[:-1]])
    mono = t > prev_max
    report['dropped_non_monotonic'] = int((~mono).sum())
    df = df[mono].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f'{path}: fewer than two valid poses')
    q = df[['qw', 'qx', 'qy', 'qz']].to_numpy(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    df[['qw', 'qx', 'qy', 'qz']] = q
    unknown = sorted(set(df['status']) - set(STATUSES))
    report['unknown_status_values'] = unknown
    report['rows_valid'] = int(len(df))
    report['status_counts_raw'] = {k: int(v) for k, v in df['status'].value_counts().items()}
    return df, report


def synthetic_trajectory(seed: int = 7, duration: float = 8.0, rate: float = 20.0) -> pd.DataFrame:
    """Human-scale handheld scoop: forward/down, drag, curl, lift, swing, dump, return.

    Timestamps are jittered, one NaN row and one backwards timestamp are
    injected, and some samples are flagged imu_only / interpolated so the
    loader and HUD are exercised.
    """
    rng = np.random.default_rng(seed)
    # t, x, y, z (m), pitch (deg, + = nose up), yaw (deg)
    key = np.array([
        [0.0, 0.00, 0.00, 0.00, 0, 0],
        [1.0, 0.28, 0.00, -0.30, -25, 0],   # reach forward and down, teeth down
        [2.2, 0.10, 0.02, -0.38, 5, 0],     # drag back through soil
        [3.2, 0.05, 0.02, -0.30, 55, 0],    # curl
        [4.4, 0.05, 0.02, 0.12, 55, 5],     # lift
        [5.6, -0.05, 0.42, 0.18, 55, 40],   # swing sideways
        [6.5, -0.05, 0.45, 0.14, -55, 40],  # dump
        [8.0, 0.00, 0.00, 0.00, 0, 0],      # return
    ], float)
    key[:, 0] *= duration / 8.0
    n = int(duration * rate) + 1
    t = np.arange(n) / rate + rng.normal(0, 0.002, n)
    t[0] = 0.0
    t = np.clip(t, 0, duration)
    interp = [PchipInterpolator(key[:, 0], key[:, i]) for i in range(1, 6)]
    x, y, z, pitch, yaw = (f(t) for f in interp)
    pos = np.c_[x, y, z] + rng.normal(0, 0.002, (n, 3))
    # Camera looks forward (+x) and 30 deg down; +x_cam = -y_world, +y_cam = down/back.
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    R0 = np.array([[0, -s, c], [-1, 0, 0], [0, -c, -s]], float)  # columns = cam axes in world
    rot = (Rotation.from_euler('z', yaw, degrees=True)
           * Rotation.from_rotvec(np.c_[np.zeros(n), -np.radians(pitch), np.zeros(n)])
           * Rotation.from_matrix(R0)
           * Rotation.from_rotvec(rng.normal(0, np.radians(0.4), (n, 3))))
    xyzw = rot.as_quat()
    status = np.array(['tracked'] * n, dtype=object)
    status[int(0.40 * n):int(0.46 * n)] = 'imu_only'
    status[int(0.66 * n):int(0.69 * n)] = 'interpolated'
    df = pd.DataFrame({'t_s': t, 'px': pos[:, 0], 'py': pos[:, 1], 'pz': pos[:, 2],
                       'qw': xyzw[:, 3], 'qx': xyzw[:, 0], 'qy': xyzw[:, 1], 'qz': xyzw[:, 2],
                       'status': status})
    df[['px', 'py', 'pz']] -= df[['px', 'py', 'pz']].iloc[0].to_numpy()
    df.loc[int(0.25 * n), 'px'] = np.nan                                 # NaN row
    glitch = df.iloc[[int(0.55 * n)]].copy(); glitch['t_s'] -= 0.5      # backwards timestamp
    df = pd.concat([df.iloc[:int(0.6 * n)], glitch, df.iloc[int(0.6 * n):]], ignore_index=True)
    return df


# --------------------------------------------------------------- preprocessing
@dataclass
class Resampled:
    t: np.ndarray            # (N,)
    pos: np.ndarray          # (N,3) camera position, world
    rot: Rotation            # N rotations world<-camera
    status: np.ndarray       # (N,) str


def resample(df: pd.DataFrame, rate: float = CONTROL_HZ, smooth_sigma: float = 1.0,
             gap_s: float | None = None) -> Resampled:
    """Resample to ``rate`` Hz (linear position, slerp orientation), Gaussian-smooth
    position by ``smooth_sigma`` samples, and mark samples in gaps as interpolated."""
    t_src = df['t_s'].to_numpy(float)
    t = t_src[0] + np.arange(int(np.floor((t_src[-1] - t_src[0]) * rate + 1e-9)) + 1) / rate
    pos = np.stack([np.interp(t, t_src, df[c].to_numpy(float)) for c in ('px', 'py', 'pz')], 1)
    q = df[['qx', 'qy', 'qz', 'qw']].to_numpy(float)
    rot = Slerp(t_src, Rotation.from_quat(q))(np.clip(t, t_src[0], t_src[-1]))
    if smooth_sigma > 0:
        pos = gaussian_filter1d(pos, smooth_sigma, axis=0, mode='nearest')
    idx = np.clip(np.searchsorted(t_src, t), 1, len(t_src) - 1)
    nearest = np.where(np.abs(t_src[idx - 1] - t) <= np.abs(t_src[idx] - t), idx - 1, idx)
    status = df['status'].to_numpy(object)[nearest].astype(str)
    gap_s = gap_s if gap_s is not None else 1.5 / rate
    status = np.where(np.abs(t_src[nearest] - t) > gap_s, 'interpolated', status)
    return Resampled(t - t[0], pos, rot, status)


def parse_mount(values) -> tuple[np.ndarray, Rotation]:
    """camera->bucket transform: tx ty tz [qw qx qy qz] (bucket point pose in camera frame)."""
    v = np.asarray(values if values is not None else [0, 0, 0, 1, 0, 0, 0], float)
    if v.size == 3:
        v = np.r_[v, 1, 0, 0, 0]
    if v.size != 7:
        raise ValueError('--cam-to-bucket expects 3 or 7 numbers: tx ty tz [qw qx qy qz]')
    return v[:3], Rotation.from_quat(np.r_[v[4:7], v[3]])


MOUNT_FILE = CONFIG / 'robot.json'          # its camera_mount section


def load_mount(path=None) -> tuple[list[float], dict]:
    """Camera->bucket transform from the camera_mount section of configs/hardware/robot.json, or from a standalone
    mount JSON: ([tx ty tz qw qx qy qz], mount fields)."""
    data = json.loads(Path(path or MOUNT_FILE).read_text())
    cfg = data.get('camera_mount', data)
    t = [float(x) for x in cfg['bucket_point_in_camera_m']]
    q = [float(x) for x in cfg.get('rotation_quat_wxyz', [1, 0, 0, 0])]
    if len(t) != 3 or len(q) != 4:
        raise ValueError(f'{path or MOUNT_FILE}: need 3 translation and 4 quaternion values')
    return t + q, cfg


def resolve_mount(cli=None, mount_file=None) -> tuple[list[float], dict]:
    """CLI values override the mount file. Returns (values, info for metrics)."""
    if cli is not None:
        vals = [float(x) for x in cli]
        vals = vals + [1.0, 0.0, 0.0, 0.0] if len(vals) == 3 else vals
        return vals, {'mount_source': 'cli (--cam-to-bucket)', 'measured': None}
    path = Path(mount_file or MOUNT_FILE)
    if not path.is_file():
        return [0, 0, 0, 1, 0, 0, 0], {'mount_source': f'default identity ({path} missing)', 'measured': False}
    vals, cfg = load_mount(path)
    return vals, {'mount_source': f'file ({path})', 'measured': bool(cfg.get('measured', False)),
                  'camera_side': cfg.get('camera_side'), 'camera_faces': cfg.get('camera_faces')}


def apply_mount(res: Resampled, mount) -> tuple[np.ndarray, Rotation]:
    t_cb, R_cb = mount
    return res.pos + res.rot.apply(t_cb), res.rot * R_cb


# -------------------------------------------------------------------- retarget
def principal_direction(P: np.ndarray, align_with: np.ndarray | None = None) -> np.ndarray:
    """Horizontal PCA axis of the lower half of the path (where the bucket works).

    Fallback only: the axis sign is taken from ``align_with`` (camera forward) when given.
    Without it the sign falls back to start -> deepest point, which is unreliable when that
    offset is small (it flipped 7 of 8 real episodes); prefer :func:`camera_heading`.
    """
    z = P[:, 2]
    low = P[z <= np.median(z)]
    H = low[:, :2] - low[:, :2].mean(0)
    if len(low) < 3 or np.allclose(H, 0):
        H = P[:, :2] - P[:, :2].mean(0)
    if np.allclose(H, 0):
        return np.array([1.0, 0, 0])
    d = np.linalg.svd(H, full_matrices=False)[2][0]
    ref = (np.asarray(align_with, float)[:2] if align_with is not None and np.linalg.norm(np.asarray(align_with)[:2]) > 1e-9
           else P[np.argmin(z), :2] - P[0, :2])
    if np.dot(ref, d) < 0:
        d = -d
    return np.r_[d, 0] / np.linalg.norm(d)


def camera_heading(rot: Rotation, status=None, vertical_deg: float = 15.0, vertical_frac: float = 0.5) -> dict:
    """Mean horizontal direction the camera looks along (world frame), from orientation only.

    Uses the optical axis R_wc @ [0,0,1] projected onto xy and averaged over ``tracked``
    samples; averaging the unnormalised projections weights each sample by how horizontal
    the axis is. If the axis is within ``vertical_deg`` of vertical for more than
    ``vertical_frac`` of the samples (camera looking straight down), falls back to image-up,
    -(R_wc @ [0,1,0]) projected onto xy. Returns {'heading': unit 3-vector or None, 'source', ...}.
    """
    R = rot.as_matrix().reshape(-1, 3, 3)
    mask = np.ones(len(R), bool)
    if status is not None:
        st = np.asarray(status).astype(str)
        if (st == 'tracked').sum() >= 3:
            mask = st == 'tracked'
    R = R[mask]
    hz = R[:, :2, 2]
    horiz = np.linalg.norm(hz, axis=1)
    near_vertical = float(np.mean(horiz < np.sin(np.radians(vertical_deg))))
    info = {'samples_used': int(mask.sum()), 'optical_axis_near_vertical_fraction': near_vertical,
            'optical_axis_mean_horizontal_norm': float(horiz.mean())}
    if near_vertical > vertical_frac:
        h = (-R[:, :2, 1]).mean(0)
        info['source'] = 'camera_image_up (optical axis near vertical)'
    else:
        h = hz.mean(0)
        info['source'] = 'camera_optical_axis'
    n = float(np.linalg.norm(h))
    info['mean_vector_norm'] = n
    info['heading'] = None if n < 1e-3 else np.r_[h / n, 0.0]
    return info


def heading_info(P: np.ndarray, rot: Rotation | None = None, status=None, mode: str = 'backhoe') -> dict:
    """Retarget heading: the world horizontal direction mapped to model +x. ``mode``:

    * 'backhoe' (default): camera forward -> model -x. The handheld camera looks along the
      bucket teeth, and on the twin the teeth point toward the cab (``tooth_tip`` sits at
      x = -0.028 on the bucket body), so scooping toward the soil is a pull toward the machine;
    * 'camera': camera forward -> model +x (teeth would point away from the cab);
    * 'pca': path PCA axis signed to agree with camera forward, -> model +x.

    Because the retarget is a pure yaw, the bucket pitch axis (``pitch_track``) is derived
    from the same vector and a physical curl stays a curl on the twin."""
    if mode not in ('camera', 'pca', 'backhoe'):
        raise ValueError("heading must be 'camera', 'pca' or 'backhoe'")
    cam = camera_heading(rot, status) if rot is not None else {'heading': None, 'source': 'none'}
    ch = cam['heading']
    if mode == 'pca' or ch is None:
        h = principal_direction(P, align_with=ch)
        src = 'pca' + ('' if mode == 'pca' else ' (fallback: camera heading degenerate)') \
            + (' signed by camera forward' if ch is not None else ' signed by start->deepest (unreliable)')
    else:
        h = ch.copy()
        src = cam['source']
    if mode == 'backhoe':
        h = -h; src += ' + 180 deg (backhoe)'
    return {'heading_world': h, 'heading_source': src, 'yaw_rad': float(-np.arctan2(h[1], h[0])),
            'heading_mode': mode, 'camera': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cam.items()}}


def phi0_window(env: Envelope, pts: np.ndarray, theta: np.ndarray, chain=None) -> np.ndarray:
    """Boolean mask over ``env.phi``: offsets phi0 for which phi0 + theta_k is an achievable
    bucket pitch at every retargeted point (per-cell closed-form feasibility)."""
    i, k = env.cells(pts)
    dphi = env.phi[1] - env.phi[0]
    J = np.arange(len(env.phi))[None, :] + np.round(theta / dphi).astype(int)[:, None]
    valid = (J >= 0) & (J < len(env.phi))
    F = np.zeros(J.shape, bool)
    I = np.broadcast_to(i[:, None], J.shape); K = np.broadcast_to(k[:, None], J.shape)
    F[valid] = env.phi_ok[I[valid], K[valid], J[valid]]
    window = F.all(0)
    if chain is not None and window.any():
        # The occupancy grid checks cell centres. Validate candidate angles at
        # the actual points too, especially beside joint limits.
        offsets = np.flatnonzero(window)
        xs, z, gap, slew_ok = env.to_plane(pts)
        _, branches = chain.ik(xs[:, None], z[:, None], theta[:, None] + env.phi[offsets])
        window[offsets] &= (branches.any(-1) & slew_ok[:, None] & (gap[:, None] <= 1e-9)).all(0)
    return window


def clip_pitch_targets(env: Envelope, pts: np.ndarray, theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Saturate the bucket pitch where no single offset fits the whole path (``--pitch-clip``).

    Picks the offset phi0 that needs the least clipping (sum of squares over samples), then moves each
    sample's phi0 + theta onto the nearest pitch achievable at its retargeted point (the per-cell
    ``env.phi_ok``), i.e. saturates it at the joint limits. Samples outside the envelope are left
    unchanged. Returns (phi0, clipped pitch targets, clip amount per sample) [rad]."""
    i, k = env.cells(pts)
    ok = env.phi_ok[i, k] & env.contains(pts)[:, None]            # (N, P) achievable pitch bins
    P, dphi = len(env.phi), env.phi[1] - env.phi[0]
    idx = np.arange(P)
    # nearest achievable bin for every bin, per sample (rows without any keep their own bin)
    near = np.tile(idx, (len(pts), 1))
    for n in np.flatnonzero(ok.any(1)):
        good = np.flatnonzero(ok[n])
        pos = np.clip(np.searchsorted(good, idx), 1, len(good) - 1) if len(good) > 1 else np.zeros(P, int)
        lo, hi = good[pos - 1] if len(good) > 1 else good[pos], good[pos]
        near[n] = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
    shift = np.round(theta / dphi).astype(int)
    best, best_cost = None, np.inf
    for j0 in range(P):
        J = np.clip(j0 + shift, 0, P - 1)
        cost = float(np.sum(((near[np.arange(len(pts)), J] - J) * dphi) ** 2)
                     + np.sum(((j0 + shift - J) * dphi) ** 2))   # leaving the pitch grid counts too
        if cost < best_cost:
            best, best_cost = j0, cost
    phi0 = float(env.phi[best])
    phi_t = phi0 + theta
    J = np.clip(best + shift, 0, P - 1)
    nearest = near[np.arange(len(pts)), J]
    # Only samples outside the achievable bins move, onto the nearest achievable bin.
    clipped = np.where((nearest != J) | (best + shift != J), env.phi[nearest], phi_t)
    return phi0, clipped, np.abs(clipped - phi_t)


def contact_mask(P: np.ndarray, height: float) -> np.ndarray:
    """Samples within ``height`` m (world z) of the path's lowest point: the bucket is in the dirt
    (``--contact-height``)."""
    return P[:, 2] <= P[:, 2].min() + height


def saturate_pitch(env: Envelope, pts: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Each pitch moved onto the nearest achievable bucket pitch at its point (env.phi_ok); points
    outside the envelope keep theirs."""
    i, k = env.cells(pts)
    ok = env.phi_ok[i, k] & env.contains(pts)[:, None]
    dphi = env.phi[1] - env.phi[0]
    J = np.clip(np.round((phi - env.phi[0]) / dphi).astype(int), 0, len(env.phi) - 1)
    out = np.array(phi, float)
    for n in np.flatnonzero(ok.any(1)):
        if not ok[n, J[n]]:
            good = np.flatnonzero(ok[n])
            out[n] = env.phi[good[np.argmin(np.abs(good - J[n]))]]
    return out


def _window_centre(env: Envelope, w: np.ndarray) -> float | None:
    """Centre of the longest run of feasible phi0 values."""
    if not w.any():
        return None
    idx = np.flatnonzero(w)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    r = max(runs, key=len)
    return float(env.phi[r].mean())


def default_yaw(P: np.ndarray, rot: Rotation | None = None, status=None, heading: str = 'backhoe') -> float:
    """Yaw mapping the heading (see :func:`heading_info`) onto model +x."""
    return heading_info(P, rot, status, heading)['yaw_rad']


def fit_retarget(P: np.ndarray, env: Envelope, yaw: float | None = None, scale: float | None = None,
                 fill: float = 0.9, theta: np.ndarray | None = None, *, pitch_for_targets=None,
                 pitch_chain=None, pitch_mask: np.ndarray | None = None) -> dict:
    """Similarity transform p_model = s * Rz(yaw) (p - c_world) + c_model fitting P inside ``env``.

    If ``theta`` (relative bucket pitch per sample) is given, the fit also requires that a
    single pitch offset phi0 makes phi0 + theta achievable along the whole path; if no
    scale satisfies that, it falls back to a position-only fit (``pitch_feasible`` False).
    ``pitch_for_targets`` evaluates pitch in the moving arm plane for each candidate
    placement. Its result must also be used for the final IK targets.
    ``pitch_mask``: only these samples need the bucket pitch (contact with the dirt); the others
    constrain position only.
    """
    m = slice(None) if pitch_mask is None else np.asarray(pitch_mask, bool)
    if yaw is None:
        yaw = default_yaw(P)
    Rz = Rotation.from_euler('z', yaw).as_matrix()
    c_world = 0.5 * (P.min(0) + P.max(0))
    Q = (P - c_world) @ Rz.T
    dist = distance_transform_edt(env.grid)
    order = np.argsort(dist.ravel())[::-1]
    cands = []
    for flat in order[:6000]:
        if dist.flat[flat] <= 0:
            break
        i, k = np.unravel_index(flat, env.grid.shape)
        c = np.array([env.x0 + (i + .5) * env.cell, env.y0, env.z0 + (k + .5) * env.cell])
        if all(np.linalg.norm(c - o) > 0.012 for o in cands):
            cands.append(c)
        if len(cands) >= 80:
            break
    extent = max(np.ptp(Q, 0).max(), 1e-6)

    def ok(s, c, pitch):
        pts = s * Q + c
        if not env.contains(pts).all():
            return False
        angles = pitch_for_targets(pts) if pitch and pitch_for_targets is not None else theta
        return (not pitch) or bool(phi0_window(env, pts[m], angles[m], pitch_chain).any())

    def search(pitch):
        best_s, best_c = -1.0, None
        for c in cands:
            if not ok(1e-9, c, pitch):
                continue
            lo, hi = 0.0, 1.0 / extent
            for _ in range(26):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ok(mid, c, pitch) else (lo, mid)
            if lo > best_s:
                best_s, best_c = lo, c
        return best_s, best_c

    use_pitch = (theta is not None or pitch_for_targets is not None) and env.phi_ok is not None
    pitch_feasible = False
    if scale is None:
        best_s, best_c = search(True) if use_pitch else (-1.0, None)
        pitch_feasible = best_c is not None and best_s > 0
        pos_s, pos_c = search(False)
        if not pitch_feasible:
            best_s, best_c = pos_s, pos_c
        s = fill * best_s
        while s > 0 and not ok(s, best_c, pitch_feasible):
            s *= 0.97
        max_s = best_s
    else:
        s = float(scale)
        pos_s = None
        fits = [c for c in cands if ok(s, c, use_pitch)]
        pitch_feasible = use_pitch and bool(fits)
        fits = fits or [c for c in cands if ok(s, c, False)]
        best_c = fits[0] if fits else cands[0]
        max_s = None
    A = s * Rz
    b = best_c - A @ c_world
    M = np.eye(4); M[:3, :3] = A; M[:3, 3] = b
    final_points = s * Q + best_c
    if use_pitch and pitch_for_targets is not None:
        theta = pitch_for_targets(final_points)
    window = phi0_window(env, final_points[m], theta[m], pitch_chain) if use_pitch else None
    phi0 = _window_centre(env, window) if use_pitch else None
    if pitch_chain is not None and phi0 is not None:
        # Keep an angle actually checked above; a midpoint can fall between bins.
        phi0 = float(env.phi[np.flatnonzero(window)[np.argmin(np.abs(env.phi[window] - phi0))]])
    return {'scale': float(s), 'max_fitting_scale': None if max_s is None else float(max_s),
            'max_fitting_scale_position_only': None if pos_s is None else float(pos_s),
            'fill_fraction': fill, 'pitch_feasible': bool(pitch_feasible), 'phi0_rad': phi0,
            'yaw_rad': float(yaw), 'yaw_deg': float(np.degrees(yaw)),
            'dig_direction_world': (Rz.T @ [1, 0, 0]).tolist(), 'world_center': c_world.tolist(),
            'model_center': best_c.tolist(), 'matrix_world_to_model': M.tolist(),
            'rotation': Rz.tolist(), 'translation': b.tolist(),
            'formula': 'p_model = scale * Rz(yaw) @ (p_world - world_center) + model_center'}


def apply_transform(tf: dict, P: np.ndarray) -> np.ndarray:
    M = np.array(tf['matrix_world_to_model'])
    return P @ M[:3, :3].T + M[:3, 3]


def pitch_track(rot: Rotation, dig_dir_world: np.ndarray, sigma: float = 1.0, axes=None) -> np.ndarray:
    """Pitch (nose-up +) of the bucket relative to the first sample (swing-twist decomposition,
    unwrapped), about the world horizontal axis perpendicular to the dig direction, or per
    sample about ``axes`` (N, 3) - the arm's pitch axis once the slew has turned the plane."""
    if axes is None:
        axis = np.cross(dig_dir_world, [0, 0, 1.0])  # = -lateral; maps to model -y after retarget
        axes = np.broadcast_to(axis / np.linalg.norm(axis), (len(rot), 3))
    rel = rot * rot[0].inv()
    xyzw = rel.as_quat()
    th = 2 * np.arctan2(np.sum(xyzw[:, :3] * axes, 1), xyzw[:, 3])
    th = np.unwrap((th + np.pi) % (2 * np.pi) - np.pi)
    th -= th[0]
    return gaussian_filter1d(th, sigma, mode='nearest') if sigma > 0 else th


def arm_plane_pitch(rot: Rotation, targets: np.ndarray, arm: Arm, world_to_model,
                    sigma: float = 1.0) -> np.ndarray:
    """Relative pitch about the slew-dependent arm axis at each target point."""
    d = np.asarray(targets)[:, :2] - arm.slew_origin[:2]
    u = np.sqrt(np.maximum(np.sum(d * d, axis=1) - arm.y0 ** 2, 0.0))
    slews = np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(arm.y0, u)
    axes = np.column_stack([np.sin(slews), -np.cos(slews), np.zeros(len(slews))])
    return pitch_track(rot, None, sigma, axes=axes @ np.asarray(world_to_model))


# ------------------------------------------------------------------- rendering
def render_video(arm: Arm, path, qpos, tip_path, targets, status, reach, t, title,
                 camera='overview', width=640, height=480, fps=20):
    m, d = arm.model, arm.data
    renderer = mujoco.Renderer(m, height=height, width=width)
    lookat = np.r_[0.5 * (targets.min(0) + targets.max(0))[:2] * 0.55, 0.15]
    cam = make_camera(m, camera, lookat)
    writer = open_writer(path, fps)
    try:
        for k in range(len(qpos)):
            d.qpos[arm.qadr] = qpos[k]; d.qvel[:] = 0
            d.mocap_pos[0] = targets[k]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            sc = renderer.scene
            for j in range(len(targets) - 1):
                future = j >= k
                add_segment(sc, targets[j], targets[j + 1],
                             (.2, 1, .45, .35 if future else .9) if reach[j] else (1, .25, .25, .9), 0.0012)
            for j in range(max(0, k - 60), k):
                add_segment(sc, tip_path[j], tip_path[j + 1], (1, .6, .1, 1), 0.0015)
            frame = np.ascontiguousarray(renderer.render())
            err = np.linalg.norm(tip_path[k] - targets[k]) * 1000
            lines = [title, f't = {t[k]:5.2f} s   status: {status[k]}',
                     f'target tracking error: {err:5.1f} mm   {"reachable" if reach[k] else "UNREACHABLE (nearest)"}',
                     'green: fitted target path   orange: achieved tip']
            writer.append_data(hud(frame, lines))
    finally:
        writer.close(); renderer.close()


def plot_joints(path, t, q_ik, q_cmd, q_dyn, targets, tip_kin, tip_dyn, status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = NQ
    fig, ax = plt.subplots(rows, 2, figsize=(13, 10), sharex=True)
    bad = status != 'tracked'
    for i, n in enumerate(JOINTS):
        a = ax[i, 0]
        a.plot(t, np.degrees(q_ik[:, i]), color='#999999', lw=1, label='IK (unlimited)')
        a.plot(t, np.degrees(q_cmd[:, i]), color='#1f5fbf', lw=1.6, label='commanded (rate-limited)')
        a.plot(t, np.degrees(q_dyn[:, i]), color='#e07b00', lw=1.4, ls='--', label='achieved (dynamic)')
        a.set_ylabel(f'{n} [deg]'); a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8, loc='best'); ax[0, 0].set_title('Joint angles: commanded vs achieved')
    for i, c in enumerate('xyz'):
        a = ax[i, 1]
        a.plot(t, targets[:, i] * 1000, color='#2a9d3a', lw=1.6, label='target')
        a.plot(t, tip_kin[:, i] * 1000, color='#1f5fbf', lw=1.2, label='FK kinematic')
        a.plot(t, tip_dyn[:, i] * 1000, color='#e07b00', lw=1.2, ls='--', label='dynamic')
        a.set_ylabel(f'tip {c} [mm]'); a.grid(alpha=.3)
    ax[0, 1].legend(fontsize=8, loc='best'); ax[0, 1].set_title('Tooth-tip position (model frame)')
    a = ax[3, 1]
    a.plot(t, np.linalg.norm(tip_kin - targets, axis=1) * 1000, color='#1f5fbf', label='kinematic')
    a.plot(t, np.linalg.norm(tip_dyn - targets, axis=1) * 1000, color='#e07b00', ls='--', label='dynamic')
    a.set_ylabel('tip error [mm]'); a.grid(alpha=.3); a.legend(fontsize=8)
    for a in ax.ravel():
        for k in np.flatnonzero(bad):
            a.axvspan(t[k] - .025, t[k] + .025, color='#f2c14e', alpha=.18, lw=0)
    ax[NQ - 1, 0].set_xlabel('t [s]  (shaded: non-tracked status)'); ax[3, 1].set_xlabel('t [s]')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------------- pipeline
def solve_actions(actions: dict, profile: str = 'full4', arm: Arm | None = None) -> dict:
    """The twin's joint / tip track for an actions.csv (columns t_s, tip_target_x/y/z, optional
    pitch_target): IK of each target (warm-started), then FK. Used for replays made with the
    earlier two-joint twin, whose joint columns are not the twin's; their targets were projected
    onto its fixed arm plane, and ``tip_target_lateral_m`` (when recorded) puts that offset back.
    Kinematics only. Returns {'t', 'q' (n, 4), 'tip' (n, 3), 'tip_err' (n,)}."""
    arm = arm or Arm(profile)
    targets = np.stack([np.asarray(actions['tip_target_' + a], float) for a in 'xyz'], 1)
    if 'tip_target_lateral_m' in actions:
        targets[:, 1] += np.asarray(actions['tip_target_lateral_m'], float)
    phi = actions.get('pitch_target')
    Q, _, _ = solve_ik(arm, targets, None if phi is None else np.asarray(phi, float))
    tip = np.array([arm.fk(q) for q in Q])
    return {'t': np.asarray(actions['t_s'], float), 'q': Q, 'tip': tip,
            'tip_err': np.linalg.norm(tip - targets, axis=1)}


def _stats(x):
    x = np.abs(np.asarray(x, float))
    return {'mean': float(x.mean()), 'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)),
            'max': float(x.max())}


def run(trajectory, output, profile='full4', mount=None, yaw_deg=None, scale=None, fill=0.9,
        pitch_gain=1.0, w_pitch=BUCKET_ANGLE_WEIGHT, time_scale=1.0, smooth_sigma=1.0, render=True,
        camera='overview', kinematic_video=True, view: dict | None = None, heading: str = 'backhoe',
        mount_file=None, planner: str = 'ik', hold_joints: bool = False, ramp_s: float = RAMP_S,
        w_smooth: float = SMOOTH_WEIGHT, pitch_clip: bool = False, contact_height: float | None = None) -> dict:
    """Replay with all four joints free: base slew, boom, stick and bucket.

    Explicitly setting ``hold_joints=True`` selects the legacy two-joint action
    contract. Only that mode fixes slew/stick at kinematics.locked_joints() and
    projects the path onto the fixed arm plane, reporting the dropped sideways motion.
    """
    if planner == 'coupled':             # coupled channels: pulse-space planner instead of IK (coupled.py)
        from .coupled import run_coupled
        return run_coupled(trajectory, output, mount, yaw_deg, scale, fill, time_scale, smooth_sigma, render,
                           heading, mount_file)
    if planner not in ('ik', 'lookahead'):
        raise ValueError('Unknown replay planner')
    if not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError('time_scale must be finite and positive')
    if planner == 'lookahead' and (ramp_s <= 0 or w_smooth != 0):
        raise ValueError('Lookahead uses hard acceleration limits: ramp_s > 0 and w_smooth = 0 required')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    df, load_rep = load_trajectory(trajectory)
    source_origin = float(df['t_s'].iloc[0])
    df = df.copy(); df['t_s'] = (df['t_s'] - df['t_s'].iloc[0]) * time_scale
    res = resample(df, CONTROL_HZ, smooth_sigma)
    dt = 1.0 / CONTROL_HZ
    mount_vals, mount_info = resolve_mount(mount, mount_file)
    if mount_info.get('measured') is False:
        print(f"WARNING: camera->bucket mount not measured ({mount_info['mount_source']}; "
              f"bucket point in camera = {mount_vals[:3]} m) - tooth-tip targets assume that offset")
    mount = mount_vals
    P_world, R_bucket = apply_mount(res, parse_mount(mount))
    arm = Arm(profile)
    locked = locked_joints() if hold_joints else None
    env = reachable_envelope(arm, locked=locked)
    hinfo = heading_info(P_world, res.rot, res.status, heading)
    if yaw_deg is None:
        yaw = hinfo['yaw_rad']
    else:
        yaw = float(np.radians(yaw_deg)); hinfo['heading_source'] = f'--yaw-deg {yaw_deg:g} override'
    # The pitch axis is derived from the same vector that maps to model +x: world axis
    # (dig_dir x z) -> model (x x z) = -y, the pitch joints' axis at slew 0. The retarget is a
    # pure yaw, so a physical bucket rotation maps to the same rotation on the twin: with the
    # backhoe heading the teeth point toward the cab and curling the scoop closed (teeth
    # swinging up/toward the cab) gives a negative pitch change, i.e. a negative bucket joint
    # (axis 0 -1 0; +bucket swings the tooth tip away from the stick = open/dump).
    dig_dir = Rotation.from_euler('z', yaw).inv().apply([1.0, 0, 0])
    hinfo['heading_world'] = dig_dir
    theta = pitch_gain * pitch_track(R_bucket, dig_dir, smooth_sigma)
    rotation = Rotation.from_euler('z', yaw).as_matrix()
    pitch_for_targets = None if hold_joints else lambda pts: pitch_gain * arm_plane_pitch(
        R_bucket, pts, arm, rotation, smooth_sigma)
    # --contact-height: the bucket angle only has to match while the bucket is in the dirt.
    contact = None if contact_height is None else contact_mask(P_world, contact_height)
    tf = fit_retarget(P_world, env, yaw, scale, fill, theta, pitch_for_targets=pitch_for_targets,
                      pitch_chain=None if hold_joints else PlanarChain(arm, joint_margin=0.03), pitch_mask=contact)
    targets = apply_transform(tf, P_world)
    gap = env.lateral(targets)       # sideways part the arm cannot reach (see Envelope.to_plane)
    if hold_joints:
        targets[:, 1] = env.y0                   # slew held: project onto the fixed arm plane
    else:
        # The slew turns the arm plane to each target, and with it the pitch axis: measure the
        # human pitch about that axis (model Rz(slew) -y, mapped back to world).
        theta = pitch_for_targets(targets)
    inside = env.contains(targets)

    clip = np.zeros(len(theta))
    saturated = np.zeros(len(theta))
    if tf['phi0_rad'] is not None and tf['pitch_feasible']:
        phi0 = tf['phi0_rad']
        phi_t = phi0 + theta
        if contact is not None:
            # Out of the dirt the angle follows the hand as far as the bucket can (saturated).
            sat = np.where(contact, phi_t, saturate_pitch(env, targets, phi_t))
            saturated, phi_t = np.abs(sat - phi_t), sat
        q_seed = None if hold_joints else analytic_seed(arm, env, targets[0], phi_t[0])
    elif pitch_clip and not hold_joints and env.phi_ok is not None:
        # No single offset fits: saturate the out-of-range samples at the achievable pitch instead.
        phi0, phi_t, clip = clip_pitch_targets(env, targets, theta)
        q_pos, _, _ = solve_ik(arm, targets, phi_t, locked=locked)
        q_seed = q_pos[0]
    else:
        # Position-only IK pass gives a natural pitch offset when no single offset is feasible.
        q_pos, _, _ = solve_ik(arm, targets, None, locked=locked)
        phi0 = float(np.mean(arm.pitch(q_pos)) - np.mean(theta))
        q_seed = q_pos[0]
        phi_t = phi0 + theta
    q_ik, pos_res, phi_res = solve_ik(arm, targets, phi_t, q_init=q_seed, w_pitch=w_pitch, locked=locked)
    reach = (pos_res <= 2e-3) & inside
    # What the arm can follow: speed-limited IK from tick to tick, keeping the bucket angle over tip
    # position (w_pitch m/rad) where the joints cannot keep up. rate_limit only reports how far the
    # unconstrained IK exceeds the speed limits.
    _, limited, rate_rep = rate_limit(arm, q_ik, dt)
    t, source_t, status = res.t, res.t.copy(), res.status
    timing = None
    if planner == 'lookahead':
        from .trajectory_ik import plan_ik_trajectory
        plan = plan_ik_trajectory(arm, targets, phi_t, res.t, reference_q=q_ik, locked=locked,
                                  w_pitch=w_pitch, ramp_s=ramp_s, dt=dt)
        t, source_t, q_cmd, timing = plan.time, plan.source_time, plan.q, plan.metrics
        targets = np.column_stack([np.interp(source_t, res.t, targets[:, j]) for j in range(3)])
        phi_t = np.interp(source_t, res.t, phi_t)
        sample = np.clip(np.searchsorted(res.t, source_t, side='right') - 1, 0, len(res.t) - 1)
        status, limited = res.status[sample], limited[sample]
        q_ik = q_cmd.copy()  # The geometric path evaluated on the new clock.
        pos_res = np.linalg.norm(np.array([arm.fk(q) for q in q_ik]) - targets, axis=1)
        phi_res = phi_t - pitch_of(q_ik)
        inside = env.contains(targets)
        reach = (pos_res <= 2e-3) & inside
    else:
        q_cmd = track_ik(arm, targets, phi_t, q_ik[0], dt, w_pitch, w_smooth, ramp_s, locked)

    tip_kin, pitch_kin, bucket_kin = propagate_kinematic(arm, q_cmd)
    tip_ik, _, _ = propagate_kinematic(arm, q_ik)
    dyn = propagate_dynamic(arm, q_cmd)

    # ---- CSVs
    pd.DataFrame({'t_s': t, 'source_t_s': source_origin + source_t / time_scale,
                  **{f'q_{n}': q_cmd[:, i] for i, n in enumerate(JOINTS)},
                  'tip_target_x': targets[:, 0], 'tip_target_y': targets[:, 1], 'tip_target_z': targets[:, 2],
                  'pitch_target': phi_t, 'reachable': reach.astype(int), 'status': status,
                  'rate_limited': limited.any(1).astype(int),
                  **{f'q_{n}_ik': q_ik[:, i] for i, n in enumerate(JOINTS)},
                  'ik_pos_residual_m': pos_res, 'ik_pitch_residual_rad': phi_res}
                 ).to_csv(out / 'actions.csv', index=False, float_format='%.6f')
    pd.DataFrame({'t_s': t, **{f'q_{n}': q_cmd[:, i] for i, n in enumerate(JOINTS)},
                  'tip_x': tip_kin[:, 0], 'tip_y': tip_kin[:, 1], 'tip_z': tip_kin[:, 2],
                  'bucket_pitch': pitch_kin,
                  'bucket_body_x': bucket_kin[:, 0], 'bucket_body_y': bucket_kin[:, 1], 'bucket_body_z': bucket_kin[:, 2],
                  'tip_err_m': np.linalg.norm(tip_kin - targets, axis=1)}
                 ).to_csv(out / 'fk_kinematic.csv', index=False, float_format='%.6f')
    pd.DataFrame({'t_s': t, 'sim_time_s': dyn['time'] - dt,
                  **{f'ctrl_{n}': dyn['ctrl'][:, i] for i, n in enumerate(JOINTS)},
                  **{f'qpos_{n}': dyn['qpos'][:, i] for i, n in enumerate(JOINTS)},
                  **{f'qvel_{n}': dyn['qvel'][:, i] for i, n in enumerate(JOINTS)},
                  'tip_x': dyn['tip'][:, 0], 'tip_y': dyn['tip'][:, 1], 'tip_z': dyn['tip'][:, 2],
                  'tip_err_m': np.linalg.norm(dyn['tip'] - targets, axis=1), 'ncon': dyn['ncon']}
                 ).to_csv(out / 'fk_dynamic.csv', index=False, float_format='%.6f')

    # ---- metrics
    e_dyn_tip = np.linalg.norm(dyn['tip'] - tip_kin, axis=1)
    e_dyn_q = dyn['qpos'] - q_cmd
    metrics = {
        'input': str(trajectory), 'input_sha256': load_rep['input_sha256'], 'profile': profile, 'control_hz': CONTROL_HZ, 'frameskip': FRAMESKIP,
        'time_scale': time_scale, 'samples': int(len(t)), 'duration_s': float(t[-1]),
        'planner': planner, 'timing': timing,
        'source_time': {'column': 'source_t_s', 'units': 'seconds in the input trajectory.csv',
                        'nonuniform': planner == 'lookahead',
                        'note': 'Use this column to align recording images; t_s is the robot execution clock.'},
        'loader': load_rep,
        'status_counts_resampled': {s: int((status == s).sum()) for s in np.unique(status)},
        'camera_to_bucket': {'t': parse_mount(mount)[0].tolist(), 'q_wxyz': np.roll(parse_mount(mount)[1].as_quat(), 1).tolist()},
        'mount_source': mount_info['mount_source'],
        'mount': {**mount_info, 'bucket_point_in_camera_m': list(mount_vals[:3]),
                  'rotation_quat_wxyz': list(mount_vals[3:]),
                  'frame': 'camera, OpenCV: +x image-right, +y down, +z optical axis'},
        'heading_source': hinfo['heading_source'], 'heading_world': np.asarray(hinfo['heading_world']).tolist(),
        'heading_world_note': 'world horizontal direction mapped to model +x (backhoe: = -camera forward)',
        'camera_forward_world': (None if hinfo['camera'].get('heading') is None
                                 else list(hinfo['camera']['heading'])),
        'yaw_deg': float(np.degrees(yaw)), 'heading_mode': heading, 'heading_camera_stats': hinfo['camera'],
        'retarget': tf,
        'joints': JOINTS, 'held_joints': locked,
        'active_joints': [name for name in JOINTS if name not in (locked or {})],
        'joint_motion_deg': {name: float(np.degrees(np.ptp(q_cmd[:, i]))) for i, name in enumerate(JOINTS)},
        'envelope': {'arm_plane_x_range_m': [float(env.samples[:, 0].min()), float(env.samples[:, 0].max())],
                     'z_range_m': [float(env.samples[:, 1].min()), float(env.samples[:, 1].max())],
                     'tip_lateral_offset_y0_m': env.y0, 'slew_range_rad': list(env.slew_range), 'cell_m': env.cell},
        'targets_inside_envelope': int(inside.sum()),
        'slew_used_deg': [float(np.degrees(q_cmd[:, 0].min())), float(np.degrees(q_cmd[:, 0].max()))],
        'unreachable_sideways_m': {**_stats(gap), 'note': (
            'retargeted path offset from the arm plane at slew 0, dropped before IK (slew held)' if hold_joints else
            'target closer to the slew axis than the arm plane offset |y0|')},
        'human_path_extent_m': np.ptp(P_world, 0).tolist(),
        'model_path_extent_m': np.ptp(targets, 0).tolist(),
        'contact': None if contact is None else {
            'height_m': contact_height, 'start_s': float(res.t[np.argmax(contact)]),
            'fraction': float(contact.mean()),
            'saturated_fraction': float((saturated > 1e-9).mean()),
            'saturated_deg': _stats(np.degrees(saturated[saturated > 1e-9])) if saturated.any() else None,
            'note': 'bucket pitch enforced (fit + IK) on samples within height_m of the lowest point (in '
                    'the dirt); elsewhere it follows the hand saturated at the achievable pitch'},
        'pitch_clip': {'enabled': bool(pitch_clip), 'applied': bool(clip.any()),
                       'fraction': float((clip > 1e-9).mean()) if len(clip) else 0.0,
                       'max_deg': float(np.degrees(clip.max())) if len(clip) else 0.0,
                       'mean_deg_clipped': float(np.degrees(clip[clip > 1e-9].mean())) if clip.any() else 0.0,
                       'note': 'source-clock samples whose bucket pitch target was saturated at the achievable '
                               'pitch (no single offset fits the path); 0 when the fit is feasible'},
        'pitch': {'offset_phi0_rad': phi0, 'gain': pitch_gain, 'weight_m_per_rad': w_pitch,
                  'single_offset_feasible_along_path': tf['pitch_feasible'],
                  'target_range_deg': [float(np.degrees(phi_t.min())), float(np.degrees(phi_t.max()))],
                  'human_pitch_range_deg': [float(np.degrees(theta.min())), float(np.degrees(theta.max()))]},
        'ik': {'position_residual_m': _stats(pos_res), 'pitch_residual_deg': _stats(np.degrees(phi_res)),
               'unreachable_count': int((~reach).sum()),
               'fk_check_max_err_m': float(np.abs(np.linalg.norm(tip_ik - targets, axis=1) - pos_res).max())},
        'smoothing': {'ramp_s': ramp_s, 'w_smooth': w_smooth,
                      'commanded_accel_rms_rad_s2': float(np.sqrt(np.mean((np.diff(q_cmd, 2, axis=0) / dt ** 2) ** 2)))
                      if len(q_cmd) > 2 else 0.0},
        'joint_rate': {**rate_rep, 'note': 'speeds the geometric IK would need on the original clock; commanded motion uses the selected planner'},
        'bucket_angle_error_deg': {
            'commanded': _stats(np.degrees(pitch_of(q_cmd) - phi_t)),
            'dynamic': _stats(np.degrees(pitch_of(dyn['qpos']) - phi_t)),
            'note': 'twin bucket pitch vs the UMI pitch target (pitch_target in actions.csv)'},
        'kinematic': {'tip_error_vs_target_m': _stats(np.linalg.norm(tip_kin - targets, axis=1))},
        'dynamic': {'unstable': bool(dyn['unstable']),
                    'ticks_with_contacts': int((dyn['ncon'] > 0).sum()), 'max_contacts': int(dyn['ncon'].max()),
                    'tip_rmse_vs_commanded_fk_m': float(np.sqrt(np.mean(e_dyn_tip ** 2))),
                    'tip_error_vs_target_m': _stats(np.linalg.norm(dyn['tip'] - targets, axis=1)),
                    'tip_rmse_vs_target_m': float(np.sqrt(np.mean(np.sum((dyn['tip'] - targets) ** 2, 1)))),
                    'joint_rmse_rad': {n: float(np.sqrt(np.mean(e_dyn_q[:, i] ** 2))) for i, n in enumerate(JOINTS)},
                    'joint_max_abs_err_rad': {n: float(np.abs(e_dyn_q[:, i]).max()) for i, n in enumerate(JOINTS)},
                    'note': 'achieved state is sampled after each 50 ms tick and compared to that tick\'s command'},
    }
    plot_joints(out / 'joints.png', t, q_ik, q_cmd, dyn['qpos'], targets, tip_kin, dyn['tip'], status)
    if render:
        render_video(arm, out / 'replay.mp4', dyn['qpos'], dyn['tip'], targets, status, reach, t,
                     'DYNAMIC replay (actuated twin)', camera)
        if kinematic_video:
            render_video(arm, out / 'replay_kinematic.mp4', q_cmd, tip_kin, targets, status, reach, t,
                         'KINEMATIC replay (FK of actions)', camera)
    metrics['outputs'] = sorted(p.name for p in out.iterdir())
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    if view:
        _view(arm, view, t, q_cmd, tip_kin, dyn, targets)
    return metrics


def _view(arm, view, t, q_cmd, tip_kin, dyn, targets):
    from .viewer import play
    mode = view.get('mode', 'dynamic')
    if mode == 'kinematic':
        tt, qq, tip = t, q_cmd, tip_kin
    elif mode == 'dynamic':
        tt, qq, tip = t, dyn['qpos'], dyn['tip']
    else:  # pulses: compile what the hardware would receive and simulate it
        from .commands import SimPulseBackend, compile_schedule
        sched = compile_schedule(t, q_cmd, arm, chunk_s=0.5)
        for w in sched['warnings']:
            print('WARNING:', w)
        res = SimPulseBackend(arm).execute(sched)
        tt, qq, tip = res['t'], res['qpos'], res['tip']
    mocap = targets[np.clip(np.searchsorted(t, tt), 0, len(t) - 1)]
    play(arm.model, arm.qadr, tt, qq, target_path=targets, trail=tip, mocap=mocap, loop=view.get('loop', False),
         max_seconds=view.get('max_seconds'), title=f'replay ({mode})')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trajectory', type=str, help='trajectory.csv (with --synthetic: where to write the generated one)')
    p.add_argument('--output', required=True)
    p.add_argument('--profile', default='full4', choices=['full4', 'full4_calibrated'],
                   help='full4: prior ranges/speeds; full4_calibrated: measured ranges/speeds (build --calibrated)')
    p.add_argument('--planner', default='lookahead', choices=['lookahead', 'ik', 'coupled'],
                   help="lookahead: preserve the path with local slowdown and braking; ik: track at recorded timing; "
                        "coupled: button-pulse plan through the machine's coupled "
                        "channels (full4_calibrated) -> actions_pulse.csv")
    p.add_argument('--hold-joints', action='store_true',
                   help='opt into the legacy boom/bucket-only contract; default IK moves base slew, boom, stick and bucket')
    p.add_argument('--synthetic', action='store_true', help='generate a synthetic scoop trajectory first')
    p.add_argument('--cam-to-bucket', type=float, nargs='+', metavar='V',
                   help='tx ty tz [qw qx qy qz]: bucket reference point pose in the camera frame (OpenCV axes); '
                        'overrides --mount-file')
    p.add_argument('--mount-file', type=str, default=None,
                   help='camera->bucket mount JSON (default: camera_mount in configs/hardware/robot.json)')
    p.add_argument('--yaw-deg', type=float,
                   help='override the retarget yaw [deg] applied to world before mapping into the model '
                        '(default: from --heading; backhoe maps camera-forward to model -x)')
    p.add_argument('--heading', default='backhoe', choices=['backhoe', 'camera', 'pca'],
                   help="retarget heading. Camera-forward = mean optical axis (image-up when the camera looks "
                        "straight down). 'backhoe' (default): camera-forward -> model -x, because the camera looks "
                        "along the teeth and the twin's teeth point toward the cab; 'camera': camera-forward -> "
                        "model +x; 'pca': path PCA axis signed by camera forward -> model +x")
    p.add_argument('--scale', type=float, help='override uniform retarget scale')
    p.add_argument('--fill', type=float, default=0.9, help='fraction of the largest fitting scale to use')
    p.add_argument('--pitch-gain', type=float, default=1.0)
    p.add_argument('--pitch-clip', action='store_true',
                   help='when no single pitch offset fits the path, saturate out-of-range bucket pitch targets '
                        'at the achievable pitch instead of a position-only fit (metrics.json pitch_clip)')
    p.add_argument('--contact-height', type=float, default=None, metavar='M',
                   help='enforce the bucket angle only while the tool is within M metres (world z) of its lowest '
                        'point (in the dirt); elsewhere only the position sizes the retarget and the angle follows '
                        'the hand saturated at what the bucket can reach')
    p.add_argument('--pitch-weight', type=float, default=BUCKET_ANGLE_WEIGHT,
                   help='bucket-angle weight in the speed-limited IK, metres of tip error per radian (default '
                        f'{BUCKET_ANGLE_WEIGHT}: 1 deg ~ {BUCKET_ANGLE_WEIGHT * np.pi / 180 * 1000:.1f} mm); higher keeps the '
                        'bucket angle closer to the UMI at the cost of tip position')
    p.add_argument('--ramp-s', type=float, default=RAMP_S,
                   help=f'smooth motion: seconds a joint takes to reach full speed (acceleration limit; default {RAMP_S}, '
                        '0 = off). Longer is smoother but follows the UMI less closely')
    p.add_argument('--smooth-weight', type=float, default=SMOOTH_WEIGHT,
                   help='extra smoothness penalty on joint acceleration, metres per rad of per-tick velocity change '
                        f'(default {SMOOTH_WEIGHT} = off)')
    p.add_argument('--time-scale', default='1.0',
                   help="playback slow-down factor (>1 slower), or 'auto' = smallest factor that keeps "
                        "the IK joint speeds within configs/hardware/robot.json limits")
    p.add_argument('--smooth-sigma', type=float, default=1.0, help='Gaussian smoothing, in 20 Hz samples')
    p.add_argument('--camera', default='overview', help="'overview' (default), 'side', 'bucket_camera'")
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--no-kinematic-video', action='store_true')
    p.add_argument('--view', action='store_true', help='play the result in the interactive MuJoCo viewer (mjpython on macOS)')
    p.add_argument('--view-mode', default='dynamic', choices=['kinematic', 'dynamic', 'pulses'])
    p.add_argument('--loop', action='store_true', help='repeat the viewer playback until the window closes')
    p.add_argument('--view-seconds', type=float, help=argparse.SUPPRESS)   # auto-close (smoke tests)
    a = p.parse_args(argv)
    if a.view:
        from .viewer import guard
        if not guard('digital_twin.sim.replay', argv):
            raise SystemExit(2)
    out = Path(a.output)
    if a.synthetic:
        out.mkdir(parents=True, exist_ok=True)
        traj = Path(a.trajectory) if a.trajectory else out / 'synthetic_trajectory.csv'
        traj.parent.mkdir(parents=True, exist_ok=True)
        synthetic_trajectory().to_csv(traj, index=False, float_format='%.6f')
    elif not a.trajectory:
        p.error('--trajectory is required unless --synthetic')
    else:
        traj = Path(a.trajectory)
    args = (traj, out, a.profile, a.cam_to_bucket, a.yaw_deg, a.scale, a.fill, a.pitch_gain, a.pitch_weight)
    if a.planner == 'coupled':
        m = run(*args, 1.0 if a.time_scale == 'auto' else float(a.time_scale), a.smooth_sigma, not a.no_render,
                heading=a.heading, mount_file=a.mount_file, planner='coupled')
        print(json.dumps({'tip_tracking_mm': {k.removesuffix('_m'): (None if v is None else round(1000 * v, 2)) for k, v in m['tip_tracking'].items()},
                          'channel_ticks': m['channel_ticks'], 'n_pulses': m['n_pulses'],
                          'slew_range_deg': m['slew_range_deg'], 'channel_model': m['channel_model']}, indent=2))
        print(f'outputs in {out}')
        return
    if a.time_scale == 'auto' and a.planner == 'lookahead':
        time_scale = 1.0  # This planner already selects local timing from all motion limits.
    elif a.time_scale == 'auto':
        probe = run(*args, 1.0, a.smooth_sigma, False, a.camera, False, heading=a.heading, mount_file=a.mount_file,
                    hold_joints=a.hold_joints)
        ratio = max(probe['joint_rate']['max_speed_ratio_per_joint'].values())
        time_scale = max(1.0, float(np.ceil(ratio * 1.05 * 10) / 10))
        print(f'auto time scale: peak IK speed ratio {ratio:.2f} -> time_scale {time_scale:.1f}')
    else:
        time_scale = float(a.time_scale)
    view = {'mode': a.view_mode, 'loop': a.loop, 'max_seconds': a.view_seconds} if a.view else None
    m = run(*args, time_scale, a.smooth_sigma, not a.no_render, a.camera, not a.no_kinematic_video, view,
            heading=a.heading, mount_file=a.mount_file, hold_joints=a.hold_joints, ramp_s=a.ramp_s,
            w_smooth=a.smooth_weight, planner=a.planner, pitch_clip=a.pitch_clip, contact_height=a.contact_height)
    summary = {'planner': m['planner'], 'duration_s': m['duration_s'],
               'active_joints': m['active_joints'], 'joint_motion_deg': m['joint_motion_deg'],
               'heading_source': m['heading_source'], 'time_scale': m['time_scale'], 'scale': m['retarget']['scale'], 'yaw_deg': m['retarget']['yaw_deg'],
               'ik_pos_residual_max_mm': 1000 * m['ik']['position_residual_m']['max'],
               'ik_pitch_residual_p95_deg': m['ik']['pitch_residual_deg']['p95'],
               'unreachable': m['ik']['unreachable_count'],
               'rate_limited_ticks': m['joint_rate']['violating_ticks_any_joint'],
               'dyn_tip_rmse_vs_target_mm': 1000 * m['dynamic']['tip_rmse_vs_target_m'],
               'dyn_tip_rmse_vs_cmd_mm': 1000 * m['dynamic']['tip_rmse_vs_commanded_fk_m']}
    print(json.dumps(summary, indent=2))
    print(f'outputs in {out}')


if __name__ == '__main__':
    main()
