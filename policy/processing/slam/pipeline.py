"""End-to-end per-episode pipeline: calibration -> ORB-SLAM3 -> IMU batch smoothing -> outputs.

The camera/IMU calibration is configs/camera.json ``camera_imu`` unless CALIBRATION (CLI --calibration) names another
file: a configs/camera.json, or a per-dataset calibration.json from ``python -m policy.processing.camera_calibration
--natural`` (pooled fit plus a gyro bias per episode, the smoother's per-episode bias prior)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policy.data import camera

from . import calib
from .frontend import track_recording
from .fusion import ImuNoise, Problem, SlamPairs, Smoother, detect_stationary
from .pairs import build_pairs, median_flow, stationary_frames
from .recording import REPO, Fisheye, Recording, episode_ids, load

# per-episode output root: set_artifacts (CLI --out, reconstruct_orbslam --output)
ARTIFACTS: Path | None = None
# camera/IMU calibration to use instead of configs/camera.json (CLI --calibration)
CALIBRATION: Path | None = None


def set_artifacts(path=None, calibration=None) -> Path:
    """Redirect per-episode outputs (CLI --out) and optionally the calibration source (CLI --calibration)."""
    global ARTIFACTS, CALIBRATION
    if path is not None:
        ARTIFACTS = Path(path).expanduser().resolve()
    CALIBRATION = Path(calibration).expanduser().resolve() if calibration else None
    return ARTIFACTS


def calibration_path() -> Path:
    return CALIBRATION or camera.camera_path()


def tracks_for(rec: Recording, cam: Fisheye, work: Path, verbose=False):
    f = work / 'tracks' / f'{rec.id}.pkl'
    if f.exists():
        tr = pickle.loads(f.read_bytes())
        if np.array_equal(tr.frame_idx, np.flatnonzero(rec.keep)):
            return tr
    tr = track_recording(rec, cam, verbose=verbose)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(tr))
    return tr


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------

def calibrate(work: Path, episodes=None, verbose=True) -> dict:
    """Natural-feature camera/IMU fit over recordings (calib.fit): pooled, per episode and leave-one-out. Run by
    ``python -m policy.processing.camera_calibration --natural``, which writes it (calibration.json) and camera.json."""
    cam = Fisheye.load()
    episodes = episodes or episode_ids()
    per, ds = {}, []
    skipped = {}
    for e in list(episodes):
        try:
            rec = load(e)
        except Exception as ex:
            skipped[e] = repr(ex); episodes = [x for x in episodes if x != e]
            continue
        tr = tracks_for(rec, cam, work)
        ta, tb, om, _ = calib.visual_rotation_pairs(rec, tr, cam)
        d = dict(ta=ta, tb=tb, omega=om, avg=calib.GyroAverager(rec.imu_t, rec.gyro))
        try:
            if len(ta) < 20:
                raise ValueError(f'only {len(ta)} usable rotation pairs')
            r = calib.fit([d])
        except Exception as ex:
            skipped[e] = repr(ex); episodes = [x for x in episodes if x != e]
            continue
        ds.append(d)
        per[e] = r
    pooled = calib.fit(ds)
    # Leave-one-out stability of the pooled estimate
    loo = []
    for i, e in enumerate(episodes):
        r = calib.fit([d for j, d in enumerate(ds) if j != i])
        loo.append({'held_out': e, 'offset_ms': r['offset_s'] * 1e3,
                    'rotation_change_deg': calib.rotation_angle_deg(r['R_cam_imu'], pooled['R_cam_imu'])})
    # Held-out residual: apply pooled R/offset (bias refit only) to each episode
    out = {
        'method': 'Natural-feature two-view rotations (fisheye KLT, essential matrix on normalized coordinates, '
                  'frame pairs ~100 ms apart) vs. gyro integrated over the same shifted interval; robust '
                  '(soft-L1) joint fit of time offset, rotation and per-episode gyro bias.',
        'sign_convention': 'matching_imu_host_time = raw_camera_time + offset_s; omega_cam = R_cam_imu (omega_gyro - bias)',
        'pooled': {k: v for k, v in pooled.items()},
        'per_episode': {e: {'offset_ms': r['offset_s'] * 1e3, 'offset_sd_ms': r['offset_sd_s'] * 1e3,
                            'rotation_diff_to_pooled_deg': calib.rotation_angle_deg(r['R_cam_imu'], pooled['R_cam_imu']),
                            'pairs': r['pairs'], 'residual_median_dps': r['residual_median_dps'],
                            'residual_rms_inliers_dps': r['residual_rms_inliers_dps'],
                            'residual_rms_dps': r['residual_rms_dps'], 'motion_rms_dps': r['motion_rms_dps'],
                            'speed_correlation': r['speed_correlation'], 'gyro_bias_dps': r['gyro_bias_dps'][0]}
                        for e, r in per.items()},
        'leave_one_out': loo,
        'stored_old_offset_ms': 20.234616,
        'episodes_skipped': skipped,
        'notes': ['The offset absorbs mean rolling-shutter readout / exposure-centre delay (timestamps are '
                  'start of exposure, auto exposure varies).',
                  'Camera-to-IMU translation is not observable from rotation rates; it is not estimated.'],
    }
    if verbose:
        p = out['pooled']
        print(f"pooled offset {p['offset_s']*1e3:.2f} ms (formal sd {p['offset_sd_s']*1e3:.2f}), "
              f"residual median {p['residual_median_dps']:.2f} dps, pairs {p['pairs']}")
        for e, r in out['per_episode'].items():
            print(f"  {e}: offset {r['offset_ms']:.1f} ms, dR {r['rotation_diff_to_pooled_deg']:.2f} deg, "
                  f"median res {r['residual_median_dps']:.2f} dps, corr {r['speed_correlation']:.3f}")
    return out


def load_calibration():
    """(calibration, R_bc = R_imu_cam, offset_s) of calibration_path(). A configs/camera.json reads as one pooled fit
    without per-episode biases; its gyro_bias_dps is every episode's bias prior (gyro_bias_prior)."""
    c = json.loads(calibration_path().read_text())
    if 'camera_imu' in c:
        m = camera.validate(c)['camera_imu']
        c = {'source': str(calibration_path()), 'sign_convention': m['sign_convention'], 'camera_imu': m,
             'pooled': {'offset_s': m['offset_s'], 'R_cam_imu': m['R_cam_imu'],
                        'R_imu_cam': np.asarray(m['R_cam_imu']).T.tolist()},
             'per_episode': {}, 'gyro_bias_dps': m['gyro_bias_dps']}
    R_cam_imu = np.array(c['pooled']['R_cam_imu'])
    U, _, Vt = np.linalg.svd(R_cam_imu)
    R_cam_imu = U @ Vt
    return c, R_cam_imu.T, c['pooled']['offset_s']   # R_bc = R_imu_cam


def gyro_bias_prior(cal: dict, e: str) -> np.ndarray:
    """Prior of episode e's gyro bias (rad/s): its pooled per-episode fit, else the calibration's one gyro_bias_dps
    (configs/camera.json), else 0."""
    eps = list(cal['per_episode'])
    return np.deg2rad(cal['pooled']['gyro_bias_dps'][eps.index(e)] if e in eps else cal.get('gyro_bias_dps', np.zeros(3)))


# ---------------------------------------------------------------------------------------------
# SLAM-segment factors from ORB-SLAM3 maps
# ---------------------------------------------------------------------------------------------

def slam_pairs(sources, gaps=(1, 2, 4, 8), min_frames=5):
    """Relative-pose factors between tracked frames (online state OK) of the same map.

    sources: list of (name, result) where result has 'state' and 'maps' (see orbslam.parse). Every map gets its
    own scale with a weak log-scale prior (sd 3); ORB-SLAM3's inertial scale is only compared, not trusted
    (its IMU initialization here happens on near-static motion and never reaches the second inertial BA).
    Returns (SlamPairs or None, list of segment descriptors).
    """
    a, b, R, t, seg = [], [], [], [], []
    segs = []
    for name, res in sources:
        state = res['state']
        for mid, m in res['maps'].items():
            ok = np.array([f < len(state) and state[f] == 2 for f in m['frames']])
            fr = m['frames'][ok]; Rw = m['R_wc'][ok]; pw = m['p_wc'][ok]
            if len(fr) < min_frames:
                continue
            metric = bool(res.get('mode') == 'imu' and m.get('imu_initialized', 0))
            s_idx = len(segs)
            segs.append({'source': name, 'map_id': int(mid), 'frames': int(len(fr)), 'first_frame': int(fr[0]),
                         'last_frame': int(fr[-1]), 'metric_prior': metric,
                         'extent_map_units': float(np.ptp(pw, axis=0).max())})
            pos = {f: i for i, f in enumerate(fr)}
            for i, f in enumerate(fr):
                for g in gaps:
                    j = pos.get(f + g)
                    if j is None:
                        continue
                    a.append(f); b.append(f + g)
                    R.append(Rw[i].T @ Rw[j]); t.append(Rw[i].T @ (pw[j] - pw[i])); seg.append(s_idx)
    if not a:
        return None, []
    return SlamPairs(np.array(a), np.array(b), np.array(R), np.array(t), np.array(seg)), segs


def vo_sources(tr, cam, rec, R_bc, offset_s, bg0):
    """Python fallback: KLT keyframe VO segments in the same format as parsed ORB-SLAM3 results."""
    from .frontend import run_vo
    from .fusion import integrate_imu_orientation
    times = rec.cam_t[tr.frame_idx] + offset_s
    Rb = integrate_imu_orientation(rec.imu_t, rec.gyro, bg0, times)
    Rc = np.einsum('ji,njk,kl->nil', R_bc, Rb, R_bc)
    vo = run_vo(tr, cam, R_gyro_rel=np.einsum('nji,njk->nik', Rc[:-1], Rc[1:]))
    maps = {}
    for sgm in np.unique(vo.segment[vo.segment >= 0]):
        f = np.flatnonzero(vo.segment == sgm)
        maps[int(sgm)] = {'frames': f, 'R_wc': vo.R_wc[f], 'p_wc': vo.p_wc[f]}
    return {'state': np.where(vo.valid, 2, 4), 'maps': maps, 'mode': 'mono'}


# ---------------------------------------------------------------------------------------------
# Per-episode fusion
# ---------------------------------------------------------------------------------------------

@dataclass
class Gate:
    """Physical plausibility gate for a tabletop handheld scoop (soil bin -> box)."""
    max_speed_mps: float = 0.5
    max_extent_m: float = 0.5
    max_imu_whitened_rms: float = 1.5


def plausibility(c: np.ndarray, v: np.ndarray, imu_whitened_rms: float, gate: Gate = Gate()):
    """Return (ok, reasons, metrics) for camera positions c (N,3) and velocities v (N,3)."""
    speed = float(np.max(np.linalg.norm(v, axis=1))) if len(v) else 0.0
    extent = float(np.max(np.ptp(c, axis=0))) if len(c) else 0.0
    reasons = []
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(v)):
        reasons.append('non-finite state')
    if speed > gate.max_speed_mps:
        reasons.append(f'max speed {speed:.2f} m/s > {gate.max_speed_mps}')
    if extent > gate.max_extent_m:
        reasons.append(f'extent {extent:.2f} m > {gate.max_extent_m}')
    if not np.isfinite(imu_whitened_rms) or imu_whitened_rms > gate.max_imu_whitened_rms:
        reasons.append(f'IMU whitened RMS {imu_whitened_rms:.2f} > {gate.max_imu_whitened_rms}')
    return not reasons, reasons, {'max_speed_mps': speed, 'max_extent_m': extent,
                                  'imu_whitened_rms': float(imu_whitened_rms)}


# Alternative initialisations tried when the standard solve fails the gate.
RESTARTS = [('standard', dict()),
            ('speed_prior_0.3', dict(speed_prior_sd=0.3)),
            ('speed_prior_0.1', dict(speed_prior_sd=0.1)),
            ('zero_init_speed_prior_0.1', dict(speed_prior_sd=0.1, init_mode='zero'))]


def _solve_with_gate(P, verbose=0, warm=None, gate: Gate = Gate()):
    tried = []
    best = None
    for name, opts in ([('warm', {})] if warm is not None else []) + RESTARTS:
        sm = Smoother(P)
        if name == 'warm':
            sm.warm_start(warm)
        for k, v in opts.items():
            setattr(sm, k, v)
        try:
            sm.solve(verbose=verbose)
        except Exception as ex:
            tried.append({'init': name, 'error': repr(ex)})
            continue
        Rc, c = sm.camera_poses()
        rms = float(np.sqrt(np.mean(sm.residuals(sm.x, parts=True)['imu'] ** 2)))
        ok, reasons, metrics = plausibility(c, sm.v, rms, gate)
        tried.append({'init': name, 'cost': sm.cost, 'plausible': ok, 'reasons': reasons, **metrics})
        if ok and (best is None or not best[1] or sm.cost < best[0].cost):
            best = (sm, ok)
        elif best is None or (not best[1] and sm.cost < best[0].cost):
            best = (sm, ok)
        if ok and name in ('standard', 'warm'):
            break   # first-choice solution is plausible; no restart needed
    sm, ok = best
    sm.gate = {'fired': len(tried) > 1 or not tried[0].get('plausible', False), 'passed': bool(ok),
               'selected_init': [t for t in tried if t.get('cost') == sm.cost][0]['init'],
               'thresholds': gate.__dict__, 'attempts': tried}
    return sm


def fuse_episode(rec: Recording, cam: Fisheye, tr, R_bc, offset_s, bg0, slam_res=None, t_bc=np.zeros(3),
                 noise: ImuNoise = None, verbose=0, warm=None):
    noise = noise or ImuNoise()
    times = rec.cam_t[tr.frame_idx] + offset_s
    vp = build_pairs(tr, cam, times=times)
    # IMU samples: after any startup gap, covering the frames
    m = (rec.imu_t > times[0] - 0.3) & (rec.imu_t < times[-1] + 0.3)
    t = rec.imu_t[m]
    gaps = np.flatnonzero(np.diff(t) > 0.05)
    if len(gaps):
        assert t[gaps[-1] + 1] < times[0], 'IMU gap inside the kept frame range'
        sel = np.flatnonzero(m)[gaps[-1] + 1:]
    else:
        sel = np.flatnonzero(m)
    imu_t, gyro, acc = rec.imu_t[sel], rec.gyro[sel], rec.accel[sel]
    quiet = detect_stationary(imu_t, gyro, acc, bg0, times)
    flow = median_flow(tr)
    still = stationary_frames(flow, quiet)
    sp, seg_maps = slam_pairs(slam_res or [])
    P = Problem(times, imu_t, gyro, acc, R_bc, np.asarray(t_bc, float), bg0, noise=noise, vis=vp, slam=sp,
                n_seg=len(seg_maps), stationary=still,
                seg_scale_prior_sd=np.full(len(seg_maps), 3.0))
    sm = _solve_with_gate(P, verbose=verbose, warm=warm)
    return sm, vp, sp, seg_maps, still, flow


def frame_status(N, vp, sp, min_pairs=1):
    vis = np.zeros(N, int)
    for arr in (vp.a, vp.b):
        np.add.at(vis, arr, 1)
    orb = np.zeros(N, bool)
    if sp is not None:
        orb[sp.a] = True; orb[sp.b] = True
    tracked = (vis >= min_pairs) | orb
    return tracked, vis, orb


def segments(mask):
    segs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append((start, i - 1)); start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs
