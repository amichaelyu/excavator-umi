"""Build two-view visual factors from KLT tracks."""
from __future__ import annotations

import numpy as np

from .frontend import Tracks, correspondences, two_view
from .fusion import VisualPairs
from .recording import Fisheye


def build_pairs(tr: Tracks, cam: Fisheye, gaps=(1, 2, 4, 8), min_corr=30, min_inliers=25,
                min_parallax_deg=1.0, sd_rot_deg=0.3, max_rate_dps=300, times=None):
    a, b, R, t, use, sdr, sdd, info = [], [], [], [], [], [], [], []
    N = len(tr.obs)
    for k in range(N):
        for g in gaps:
            j = k + g
            if j >= N:
                break
            common, pa, pb = correspondences(tr, k, j)
            if len(common) < min_corr:
                continue
            res = two_view(cam, pa, pb)
            if res is None:
                continue
            R_ab, t_ab, inl, inf = res
            if inf['inliers'] < min_inliers:
                continue
            ang = np.linalg.norm(_rotvec(R_ab))
            if times is not None and ang / max(times[j] - times[k], 1e-3) > np.deg2rad(max_rate_dps):
                continue
            par = np.deg2rad(inf['parallax_deg'])
            a.append(k); b.append(j); R.append(R_ab); t.append(t_ab / np.linalg.norm(t_ab))
            use.append(inf['parallax_deg'] >= min_parallax_deg and inf['cheirality'] >= 0.8 * inf['inliers'])
            sdr.append(np.deg2rad(sd_rot_deg) * (1 + 0.1 * g))
            sdd.append(float(np.clip(2.0 / cam.fx / max(par, 1e-4), 0.03, 0.3)))
            info.append(inf)
    return VisualPairs(np.array(a, int), np.array(b, int), np.array(R).reshape(-1, 3, 3), np.array(t).reshape(-1, 3),
                       np.array(use, bool), np.array(sdr), np.array(sdd))


def _rotvec(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_rotvec()


def median_flow(tr: Tracks) -> np.ndarray:
    """Median pixel displacement of tracks between consecutive processed frames (inf if unknown)."""
    N = len(tr.obs)
    flow = np.full(N - 1, np.inf)
    for k in range(N - 1):
        common, pa, pb = correspondences(tr, k, k + 1)
        if len(common) >= 30:
            flow[k] = float(np.median(np.linalg.norm(pb - pa, axis=1)))
    return flow


def stationary_frames(flow: np.ndarray, imu_quiet: np.ndarray, flow_px=0.35) -> np.ndarray:
    """A frame is at rest if the image barely moves to both neighbours and the IMU is quiet."""
    N = len(flow) + 1
    still = np.zeros(N, bool)
    for k in range(N):
        f = [flow[k - 1]] if k > 0 else []
        if k < N - 1:
            f.append(flow[k])
        still[k] = bool(len(f) and max(f) < flow_px and imu_quiet[k])
    return still
