"""Fisheye-aware KLT feature tracking and monocular visual odometry (Python fallback front end).

The tracker works on the raw fisheye images, undistorts tracked pixels to normalized
coordinates with the pinhole-equidistant fisheye model and estimates two-view geometry
(essential matrix, 5-point RANSAC) on those normalized points. The 1280x720 image
has a maximum field angle of ~79 deg, so all pixels have a finite pinhole projection.

A lightweight keyframe VO (triangulate + PnP + local refinement) turns tracks into
up-to-scale camera poses, split into segments whenever tracking breaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .recording import Fisheye, Recording


@dataclass
class Tracks:
    """Feature observations: per frame arrays of (track_id, u, v)."""
    frame_idx: np.ndarray                    # recording frame indices processed
    obs: list = field(default_factory=list)  # list of (ids[int], px[N,2]) per processed frame
    brightness: np.ndarray = None
    sharpness: np.ndarray = None


def preprocess(img: np.ndarray, clahe) -> np.ndarray:
    return clahe.apply(img)


def track_recording(rec: Recording, cam: Fisheye, max_features=500, min_distance=18,
                    fb_thresh=1.0, verbose=False) -> Tracks:
    idx = np.flatnonzero(rec.keep)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lk = dict(winSize=(25, 25), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tr = Tracks(idx)
    bright, sharp = [], []
    prev = None
    ids = np.zeros(0, int); pts = np.zeros((0, 2), np.float32)
    next_id = 0
    # Ignore the extreme fisheye border where the calibration is weakest.
    h, w = cam.size[1], cam.size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    th = np.hypot((xx - cam.K[0, 2]) / cam.K[0, 0], (yy - cam.K[1, 2]) / cam.K[1, 1])
    base_mask = (th < 1.30).astype(np.uint8) * 255
    for k, i in enumerate(idx):
        raw = rec.image(i)
        bright.append(float(raw.mean()))
        sharp.append(float(cv2.Laplacian(cv2.resize(raw, (640, 360)), cv2.CV_32F).var()))
        img = preprocess(raw, clahe)
        if prev is not None and len(pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img, prev, nxt, None, **lk)
            fb = np.linalg.norm(back - pts, axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thresh)
            inb = (nxt[:, 0] >= 2) & (nxt[:, 0] < w - 2) & (nxt[:, 1] >= 2) & (nxt[:, 1] < h - 2)
            ok &= inb
            ids, pts = ids[ok], nxt[ok]
            if len(pts) >= 12:
                # Geometric outlier rejection with the essential matrix on normalized coordinates.
                prev_pts = tr.obs[-1][1]
                prev_map = {t: j for j, t in enumerate(tr.obs[-1][0])}
                p0 = prev_pts[[prev_map[t] for t in ids]]
                n0, n1 = cam.undistort(p0), cam.undistort(pts)
                E, m = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1.5 / cam.fx)
                if E is not None and m is not None:
                    keep = m.ravel().astype(bool)
                    ids, pts = ids[keep], pts[keep]
        # Replenish features away from existing ones.
        if len(pts) < max_features * 0.7:
            mask = base_mask.copy()
            for p in pts:
                cv2.circle(mask, (int(p[0]), int(p[1])), min_distance, 0, -1)
            new = cv2.goodFeaturesToTrack(img, max_features - len(pts), 0.01, min_distance, mask=mask, blockSize=7)
            if new is not None:
                new = new.reshape(-1, 2).astype(np.float32)
                cv2.cornerSubPix(img, new.reshape(-1, 1, 2), (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
                ids = np.r_[ids, np.arange(next_id, next_id + len(new))]
                next_id += len(new)
                pts = np.vstack([pts, new])
        tr.obs.append((ids.copy(), pts.copy()))
        prev = img
        if verbose and k % 50 == 0:
            print(f'  tracked frame {k}/{len(idx)}: {len(pts)} features', flush=True)
    tr.brightness = np.array(bright)
    tr.sharpness = np.array(sharp)
    return tr


def correspondences(tr: Tracks, a: int, b: int):
    """Pixel correspondences between processed frames a and b (tracks alive in both)."""
    ia, pa = tr.obs[a]; ib, pb = tr.obs[b]
    common, ja, jb = np.intersect1d(ia, ib, return_indices=True)
    return common, pa[ja], pb[jb]


def two_view(cam: Fisheye, pa: np.ndarray, pb: np.ndarray, thresh_px=1.5):
    """Relative pose of b w.r.t. a from pixel correspondences.

    Returns (R_ab, t_ab_unit, inlier_mask, info) where X_a = R_ab X_b + t_ab, i.e. T_a_b,
    or None if degenerate. info has the median parallax (deg) and inlier count.
    """
    if len(pa) < 15:
        return None
    na, nb = cam.undistort(pa), cam.undistort(pb)
    E, m = cv2.findEssentialMat(na, nb, np.eye(3), cv2.RANSAC, 0.999, thresh_px / cam.fx)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, t, m2 = cv2.recoverPose(E, na, nb, np.eye(3), mask=m.copy())
    inl = m.ravel().astype(bool)
    if inl.sum() < 12:
        return None
    # recoverPose gives x_b = R x_a + t  ->  T_b_a. Invert to T_a_b.
    R_ab = R.T
    t_ab = -R.T @ t.ravel()
    ba = np.column_stack([na, np.ones(len(na))]); bb = np.column_stack([nb, np.ones(len(nb))])
    ba /= np.linalg.norm(ba, axis=1, keepdims=True); bb /= np.linalg.norm(bb, axis=1, keepdims=True)
    rot_b = bb @ R_ab.T  # bearings of b rotated into a
    par = np.degrees(np.arccos(np.clip(np.sum(ba * rot_b, axis=1), -1, 1)))
    return R_ab, t_ab, inl, {'parallax_deg': float(np.median(par[inl])), 'inliers': int(inl.sum()),
                             'cheirality': int(n_in)}


def rotation_only(cam: Fisheye, pa, pb):
    """Rotation-only fit (Kabsch on bearings, RANSAC-free, robust reweighting)."""
    ba, bb = cam.bearings(pa), cam.bearings(pb)
    w = np.ones(len(ba))
    for _ in range(5):
        H = (bb * w[:, None]).T @ ba
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T  # R_ab: ba ~ R bb
        r = np.linalg.norm(ba - bb @ R.T, axis=1)
        s = max(np.median(r) * 1.5, 1e-4)
        w = 1 / np.maximum(1, r / s)
    return R


# ---------------------------------------------------------------------------------------------
# Keyframe monocular VO producing up-to-scale segments.
# ---------------------------------------------------------------------------------------------

def _triangulate(Ra, ta, Rb, tb, na, nb):
    """Triangulate from world->camera poses (R,t: X_c = R X_w + t) and normalized points."""
    Pa = np.hstack([Ra, ta[:, None]]); Pb = np.hstack([Rb, tb[:, None]])
    X = cv2.triangulatePoints(Pa, Pb, na.T.astype(np.float64), nb.T.astype(np.float64))
    X = (X[:3] / X[3]).T
    za = (X @ Ra.T + ta)[:, 2]; zb = (X @ Rb.T + tb)[:, 2]
    return X, (za > 0) & (zb > 0)


def _ray_angle(Ra, ta, Rb, tb, X):
    ca = -Ra.T @ ta; cb = -Rb.T @ tb
    va = X - ca; vb = X - cb
    cosang = np.sum(va * vb, 1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


@dataclass
class VOResult:
    frame_idx: np.ndarray        # recording frame index per processed frame
    valid: np.ndarray            # bool: pose available
    segment: np.ndarray          # segment id per processed frame (-1 invalid)
    R_wc: np.ndarray             # (N,3,3) camera-to-visual-world rotation
    p_wc: np.ndarray             # (N,3) camera position (arbitrary scale per segment)
    n_inliers: np.ndarray


def run_vo(tr: Tracks, cam: Fisheye, R_gyro_rel=None, min_parallax_deg=2.0, min_pnp=25, verbose=False) -> VOResult:
    """Keyframe VO. R_gyro_rel[k] (optional): gyro-predicted camera rotation R_{c_k c_{k+1}},
    used to seed PnP and to help initialization."""
    N = len(tr.obs)
    R_wc = np.tile(np.eye(3), (N, 1, 1)); p_wc = np.zeros((N, 3))
    valid = np.zeros(N, bool); seg = -np.ones(N, int); ninl = np.zeros(N, int)
    landmarks = {}   # track id -> world point (current segment)
    seg_id = -1
    k = 0
    lost_since_init = True
    while k < N - 1:
        if lost_since_init:
            # Initialize: search a later frame with enough parallax.
            init = None
            for j in range(k + 1, min(k + 15, N)):
                common, pa, pb = correspondences(tr, k, j)
                if len(common) < 40:
                    break
                res = two_view(cam, pa, pb)
                if res is None:
                    continue
                R_ab, t_ab, inl, info = res
                if info['parallax_deg'] >= min_parallax_deg and info['inliers'] >= 40:
                    init = (j, common[inl], pa[inl], pb[inl], R_ab, t_ab)
                    break
            if init is None:
                k += 1
                continue
            j, cids, pa, pb, R_ab, t_ab = init
            # world = camera k
            Ra, ta = np.eye(3), np.zeros(3)
            Rb, tb = R_ab.T, -R_ab.T @ t_ab   # world->camera b
            X, ok = _triangulate(Ra, ta, Rb, tb, cam.undistort(pa), cam.undistort(pb))
            ok &= _ray_angle(Ra, ta, Rb, tb, X) > 0.5
            if ok.sum() < 30:
                k += 1
                continue
            # Normalize scale: median depth = 1
            d = np.median(X[ok][:, 2]); X /= d; tb = tb / d
            seg_id += 1
            landmarks = {int(c): x for c, x, o in zip(cids, X, ok) if o}
            for m, (R, t) in ((k, (Ra, ta)), (j, (Rb, tb))):
                R_wc[m] = R.T; p_wc[m] = -R.T @ t; valid[m] = True; seg[m] = seg_id
            ninl[k] = ninl[j] = int(ok.sum())
            # Fill frames between k and j by PnP later in the loop
            last_kf = j
            kf_list = [k, j]
            lost_since_init = False
            m = k + 1
            while m < j:
                _pnp_frame(tr, cam, m, landmarks, R_wc, p_wc, valid, seg, ninl, seg_id, min_pnp, R_wc[m - 1], p_wc[m - 1])
                m += 1
            k = j
            if verbose:
                print(f'  VO segment {seg_id} initialized at frames {kf_list} with {len(landmarks)} points', flush=True)
            continue
        # Track frame k+1 against landmarks
        m = k + 1
        Rg = R_wc[k] @ R_gyro_rel[k] if R_gyro_rel is not None else R_wc[k]
        ok = _pnp_frame(tr, cam, m, landmarks, R_wc, p_wc, valid, seg, ninl, seg_id, min_pnp, Rg, p_wc[k])
        if not ok:
            lost_since_init = True
            k = m
            continue
        # Keyframe decision: triangulate new points when parallax to last keyframe is enough
        common, pa, pb = correspondences(tr, last_kf, m)
        if len(common):
            Ra_, ta_ = R_wc[last_kf].T, -R_wc[last_kf].T @ p_wc[last_kf]
            Rb_, tb_ = R_wc[m].T, -R_wc[m].T @ p_wc[m]
            new = np.array([c not in landmarks for c in common])
            n_tracked = sum(1 for c in tr.obs[m][0] if int(c) in landmarks)
            if new.sum() > 10 and (n_tracked < 150 or m - last_kf >= 4):
                X, okk = _triangulate(Ra_, ta_, Rb_, tb_, cam.undistort(pa[new]), cam.undistort(pb[new]))
                okk &= _ray_angle(Ra_, ta_, Rb_, tb_, X) > 1.0
                for c, x, o in zip(common[new], X, okk):
                    if o:
                        landmarks[int(c)] = x
                last_kf = m
        # Keep only the landmarks observed in this frame
        alive = set(int(c) for c in tr.obs[m][0])
        landmarks = {c: x for c, x in landmarks.items() if c in alive}
        k = m
    return VOResult(tr.frame_idx, valid, seg, R_wc, p_wc, ninl)


def _pnp_frame(tr, cam, m, landmarks, R_wc, p_wc, valid, seg, ninl, seg_id, min_pnp, R_guess, p_guess):
    ids, px = tr.obs[m]
    sel = [q for q, c in enumerate(ids) if int(c) in landmarks]
    if len(sel) < min_pnp:
        return False
    Xw = np.array([landmarks[int(ids[q])] for q in sel]); n = cam.undistort(px[sel])
    Rcw = R_guess.T; tcw = -Rcw @ p_guess
    rvec, _ = cv2.Rodrigues(Rcw)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(Xw, n, np.eye(3), None, rvec.copy(), tcw.reshape(3, 1).copy(),
                                             useExtrinsicGuess=True, iterationsCount=200,
                                             reprojectionError=2.0 / cam.fx, confidence=0.999,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < min_pnp:
        return False
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(Xw[inl], n[inl], np.eye(3), None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    R_wc[m] = R.T; p_wc[m] = (-R.T @ tvec).ravel(); valid[m] = True; seg[m] = seg_id; ninl[m] = len(inl)
    # Drop outlier landmarks
    out = np.setdiff1d(np.arange(len(sel)), inl)
    for q in out:
        landmarks.pop(int(ids[sel[q]]), None)
    return True
