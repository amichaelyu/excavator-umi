"""CLI: python -m policy.processing.slam {slam,fuse,all,summary} [episodes...] --out DIR

ORB-SLAM3 + KLT + IMU batch smoother (reconstruct.method orbslam3_smoother) per recording; outputs go to --out.

The camera/IMU calibration is configs/camera.json unless --calibration names another (pipeline.py); it is copied
into --out as calibration.json. Calibrate with python -m policy.processing.camera_calibration."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from policy.data import camera

from . import orbslam, pipeline, report
from .fusion import ImuNoise
from .pipeline import frame_status, fuse_episode, gyro_bias_prior, load_calibration, segments, tracks_for
from . import recording
from .recording import REPO, Fisheye, episode_ids, load

DEFAULT_WORK = Path(os.environ.get('UMI_SLAM_WORK', '/tmp/umi-slam'))


def slam_all(work: Path, episodes, variant, modes=('imu', 'mono'), jobs=4, rerun=False):
    cam = Fisheye.load()
    _, R_bc, off = load_calibration()
    recs = {}
    for e in episodes:
        try:
            recs[e] = load(e)
        except Exception as ex:
            print(f'{e}: cannot load for ORB-SLAM3: {ex!r}', flush=True)
    episodes = list(recs)
    todo = [(e, m) for m in modes for e in episodes]

    def go(j):
        try:
            return _go(j)
        except Exception as ex:
            print(f'{j[0]} {variant}-{j[1]}: ORB-SLAM3 FAILED {ex!r}', flush=True)
            return j, {'error': repr(ex)}

    def _go(j):
        e, m = j
        res = orbslam.run_episode(recs[e], cam, work, R_bc, np.zeros(3), off, ImuNoise(), variant, m, rerun=rerun)
        s = orbslam.summarize(res, int(recs[e].keep.sum()))
        print(f"{e} {variant}-{m}: online OK {s['online_ok']}/{s['frames']}, saved-map frames "
              f"{s['frames_in_saved_maps']}, maps {len(s['maps'])}", flush=True)
        return (e, m), s
    with ThreadPoolExecutor(jobs) as ex:
        return dict(ex.map(go, todo))


def episode_label(e):
    f = Path(recording.DATASET) / e / 'metadata.json'
    return (json.loads(f.read_text()).get('label') or '') if f.is_file() else ''


def data_quality(e, rec):
    mq = recording.manifest_quality(e)
    q = dict(rec.quality)
    flags = []
    if mq.get('needs_review'):
        flags.append('needs_review')
    if mq.get('video_recovered'):
        flags.append('video_recovered')
    if mq.get('recorder_error'):
        flags.append('recorder_error')
    if mq.get('recorder_warning'):
        flags.append('recorder_warning')
    if q.get('imu_ended_early'):
        flags.append(f"imu_ended_early(-{q['camera_time_lost_s']:.2f}s)")
    if q.get('frames_undecodable'):
        flags.append(f"undecodable_frames={q['frames_undecodable']}")
    if q.get('frames_lenient_decoded'):
        flags.append(f"lenient_decoded_frames={q['frames_lenient_decoded']}")
    if q.get('frames_csv_truncated') or q.get('imu_csv_truncated'):
        flags.append('csv_truncated')
    if q.get('imu_invalid_samples_removed'):
        flags.append(f"imu_invalid_samples={q['imu_invalid_samples_removed']}")
    return {'flags': flags, 'manifest': mq, 'loader': q}


def process(e: str, work: Path, variant: str, video=True, sensitivity=True, verbose=0,
            require_orb_slam3=False):
    t_start = time.time()
    cam = Fisheye.load()
    cal, R_bc, off = load_calibration()
    rec = load(e)
    tr = tracks_for(rec, cam, work)
    bg0 = gyro_bias_prior(cal, e)
    failures = []
    # ORB-SLAM3 results: monocular-inertial (default ORB settings; CLAHE settings made VI initialization
    # worse on this data) and camera-only (settings variant with more frames in saved maps).
    orb, orb_summ = {}, {}
    try:
        res = orbslam.run_episode(rec, cam, work, R_bc, np.zeros(3), off, ImuNoise(), 'default', 'imu')
        res['mode'] = 'imu'
        orb['imu'] = res; orb_summ['imu'] = orbslam.summarize(res, len(tr.frame_idx))
    except Exception as ex:
        failures.append(f'ORB-SLAM3 monocular-inertial failed: {ex!r}')
    best = None
    for v in orbslam.VARIANTS:
        try:
            res = orbslam.run_episode(rec, cam, work, R_bc, np.zeros(3), off, ImuNoise(), v, 'mono')
            res['mode'] = 'mono'
            sm_ = orbslam.summarize(res, len(tr.frame_idx))
            orb_summ[f'mono-{v}'] = sm_
            if require_orb_slam3:
                from .pipeline import slam_pairs
                if res.get('returncode') != 0 or slam_pairs([('candidate', res)])[0] is None:
                    continue
            if best is None or sm_['frames_in_saved_maps'] > best[1]['frames_in_saved_maps']:
                best = (res, sm_, v)
        except Exception as ex:
            failures.append(f'ORB-SLAM3 camera-only ({v}) failed: {ex!r}')
    sources = []
    if 'imu' in orb:
        sources.append(('orb_slam3_vi', orb['imu']))
    if best is not None:
        orb['mono'] = best[0]
        sources.append((f'orb_slam3_mono_{best[2]}', best[0]))
    if require_orb_slam3:
        from .pipeline import slam_pairs
        sources = [(name, result) for name, result in sources
                   if result.get('returncode') == 0 and slam_pairs([(name, result)])[0] is not None]
        if not sources:
            raise RuntimeError('ORB-SLAM3 produced no usable tracked map segment; '
                               'strict reconstruction refuses the KLT fallback. Diagnostics: '
                               + json.dumps({'runs': orb_summ, 'failures': failures}))
    if not sources:
        failures.append('ORB-SLAM3 unavailable: using the Python KLT keyframe VO fallback for scale-consistent segments')
        from .pipeline import vo_sources
        sources.append(('python_klt_vo_fallback', vo_sources(tr, cam, rec, R_bc, off, bg0)))
    slam_res = sources
    t_bc = np.zeros(3)
    sm, vp, sp, seg_maps, still, flow = fuse_episode(rec, cam, tr, R_bc, off, bg0, slam_res, t_bc, verbose=verbose)
    N = sm.N
    tracked, nvis, orbc = frame_status(N, vp, sp)
    status = np.where(tracked, 'tracked', 'imu_only')
    # Output frame
    Rc, c = sm.camera_poses()
    R_out, p0 = report.output_frame(Rc[0], c[0])
    Rc_o = np.einsum('ij,njk->nik', R_out, Rc)
    c_o = (c - p0) @ R_out.T
    v_o = sm.v @ R_out.T
    t_s = rec.cam_t[tr.frame_idx] - rec.t0
    out = pipeline.ARTIFACTS / e
    out.mkdir(parents=True, exist_ok=True)
    report.write_csv(out / 'trajectory.csv', t_s, Rc_o, c_o, status)
    ti, Ri, ci, fk = sm.imu_rate()
    Ri_o = np.einsum('ij,njk->nik', R_out, Ri); ci_o = (ci - p0) @ R_out.T
    ti_s = ti - off - rec.t0
    st_i = []
    for tt, k in zip(ti, fk):
        if abs(tt - sm.P.times[k]) < 1e-9:
            st_i.append(status[k])
        elif k + 1 < N and tracked[k] and tracked[k + 1]:
            st_i.append('interpolated')
        else:
            st_i.append('imu_only')
    report.write_csv(out / 'trajectory_imu_rate.csv', ti_s, Ri_o, ci_o, st_i)
    diag = sm.diagnostics()
    # Sensitivity of the metric scale to the IMU noise model / priors
    sens = {}
    path_len = float(np.sum(np.linalg.norm(np.diff(c_o, axis=0), axis=1)))
    extent = np.ptp(c_o, axis=0)
    if sensitivity:
        base = ImuNoise()
        variants = {'accel_noise_x0.5': replace(base, accel_nd=base.accel_nd * 0.5),
                    'accel_noise_x2': replace(base, accel_nd=base.accel_nd * 2),
                    'accel_bias_prior_x2.5': replace(base, accel_bias_prior_sd=base.accel_bias_prior_sd * 2.5),
                    'gyro_noise_x3': replace(base, gyro_nd=base.gyro_nd * 3)}
        for name, nz in variants.items():
            try:
                s2 = fuse_episode(rec, cam, tr, R_bc, off, bg0, slam_res, t_bc, noise=nz, warm=sm)[0]
                if not s2.gate['passed']:
                    sens[name] = {'error': 'no plausible solution', 'gate': s2.gate['attempts']}
                    continue
                _, c2 = s2.camera_poses()
                c2 = (c2 - c2[0]) @ R_out.T
                sens[name] = {'cost': s2.cost,
                              'path_length_m': float(np.sum(np.linalg.norm(np.diff(c2, axis=0), axis=1))),
                              'extent_m': np.ptp(c2, axis=0).tolist(),
                              'max_position_diff_m': float(np.max(np.linalg.norm(c2 - c_o, axis=1)))}
            except Exception as ex:
                sens[name] = {'error': repr(ex)}
        ratios = [v['path_length_m'] / path_len for v in sens.values() if 'path_length_m' in v]
        spread = float(max(abs(np.log(r)) for r in ratios)) if ratios else None
    else:
        spread = None
    # Scales of SLAM segments relative to the IMU-derived metric scale
    seg_info = []
    for k, sg in enumerate(seg_maps):
        seg_info.append(dict(sg, fused_scale_factor=float(np.exp(sm.logs[k])),
                             t_first_s=float(t_s[sg['first_frame']]), t_last_s=float(t_s[sg['last_frame']])))
    speed = np.linalg.norm(v_o, axis=1)
    vis_segs = segments(tracked)
    conf = 'low'
    if spread is not None and spread < 0.15 and diag['whitened_rms']['imu'] < 1.0 and sm.gate['passed'] and len(seg_maps) > 0:
        conf = 'medium'
    summary = {
        'episode': e,
        'time_origin': {'raw_camera_timestamp_ns_at_t0': int(round(rec.t0 * 1e9)),
                        'recording_started_monotonic_s': rec.meta['started_monotonic_s'],
                        'frames_dropped_at_start': int(np.argmax(rec.keep)),
                        'note': 't_s = raw camera start-of-exposure time - first kept frame time'},
        'label': episode_label(e),
        'frames_undecodable': rec.quality.get('frames_undecodable', 0),
        'data_quality': data_quality(e, rec),
        'plausibility_gate': sm.gate,
        'n_slam_segments_in_fusion': len(seg_maps),
        'frames_total': int(rec.n), 'frames_processed': int(N), 'frames_tracked': int(tracked.sum()),
        'frames_imu_only': int((~tracked).sum()), 'frames_zero_velocity': int(still.sum()),
        'tracked_segments': [[float(t_s[a]), float(t_s[b])] for a, b in vis_segs],
        'n_tracked_segments': len(vis_segs),
        'longest_imu_only_gap_s': float(max([t_s[b] - t_s[a] + 0.05 for a, b in segments(~tracked)] or [0.0])),
        'slam_front_end': {
            'sources_in_fusion': [n for n, _ in sources] + ['klt_two_view'],
            'orb_slam3_vi': orb_summ.get('imu'),
            'orb_slam3_camera_only': {k: v for k, v in orb_summ.items() if k.startswith('mono')},
            'frames_with_slam_segment_factors': int(orbc.sum()),
            'klt_two_view_pairs': int(len(vp.a)), 'klt_pairs_with_translation_direction': int(vp.use_dir.sum())},
        'orb_slam3_segments_in_fusion': seg_info,
        'calibration': {'time_offset_s': off, 'sign': cal['sign_convention'],
                        'R_imu_camera': R_bc.tolist(), 't_imu_camera_m': t_bc.tolist(),
                        't_imu_camera_note': 'camera-to-IMU lever arm not measured; assumed 0 (rigid mount, likely < 5 cm)'},
        'scale': {'source': 'IMU accelerometer via the IMU batch smoother. KLT two-view factors are '
                            'scale-free; each SLAM map segment (incl. ORB-SLAM3 inertial maps) gets its own free scale '
                            'factor (weak log-prior sd 3), so metric scale comes from the accelerometer only',
                  'confidence': conf,
                  'sensitivity_max_abs_log_path_ratio': spread, 'sensitivity': sens},
        'fusion': diag,
        'gravity_alignment_residual_deg': {'all_frames_median': diag['gravity_residual_deg_all_frames_median'],
                                           'all_frames_p95': diag['gravity_residual_deg_all_frames_p95'],
                                           'zero_velocity_frames_mean': diag['gravity_residual_deg_stationary_mean']},
        'path_length_m': path_len, 'max_speed_mps': float(speed.max()),
        'bbox_min_m': c_o.min(0).tolist(), 'bbox_max_m': c_o.max(0).tolist(), 'extent_m': extent.tolist(),
        'start_to_end_displacement_m': float(np.linalg.norm(c_o[-1] - c_o[0])),
        'failures': failures,
        'runtime_s': None,
        'pose_convention': 'T_world_camera; world gravity-aligned +z up, origin = first processed camera pose, '
                           'yaw: first optical axis projects to +x; camera frame x right, y down, z optical axis',
    }
    try:
        report.plot(out / 'trajectory.png', t_s, c_o, list(status),
                    f'{e}: {int(tracked.sum())}/{N} frames visually tracked, path {path_len:.2f} m, '
                    f'scale confidence {conf}', Rc_o)
    except Exception as ex:
        failures.append(f'plot failed: {ex!r}')
    if video:
        try:
            ost = orb['imu']['state'] if 'imu' in orb else np.full(N, -1)
            report.overlay_video(out / 'overlay.mp4', rec, tr, list(status), ost, t_s, c_o)
        except Exception as ex:
            failures.append(f'overlay failed: {ex!r}')
    # per-frame detail (small)
    with open(out / 'frames_detail.csv', 'w') as f:
        f.write('frame,t_s,status,klt_pairs,orb_vi_state,orb_vi_factor,zero_velocity,median_flow_px,brightness,sharpness\n')
        ost = orb['imu']['state'] if 'imu' in orb else np.full(N, -1)
        for k in range(N):
            fl = flow[k] if k < len(flow) else np.nan
            f.write(f'{tr.frame_idx[k]},{t_s[k]:.6f},{status[k]},{nvis[k]},{int(ost[k]) if k < len(ost) else -1},'
                    f'{int(orbc[k])},{int(still[k])},{fl:.3f},{tr.brightness[k]:.1f},{tr.sharpness[k]:.1f}\n')
    summary['runtime_s'] = time.time() - t_start
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f"{e}: tracked {int(tracked.sum())}/{N}, path {path_len:.2f} m, extent {np.round(extent, 2)}, "
          f"max speed {speed.max():.2f} m/s, scale spread {spread}, grav med {diag['gravity_residual_deg_all_frames_median']:.2f} deg",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(prog='python -m policy.processing.slam', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['slam', 'fuse', 'all', 'summary'])
    ap.add_argument('episodes', nargs='*')
    ap.add_argument('--work', type=Path, default=DEFAULT_WORK,
                    help='scratch dir for tracks, EuRoC exports and ORB-SLAM3 results (mounted as /data when ORB-SLAM3 runs through Docker)')
    ap.add_argument('--variant', default='default', choices=list(orbslam.VARIANTS),
                    help='ORB settings variant for the slam command (fuse uses default for VI and the better '
                         'camera-only variant)')
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--no-sensitivity', action='store_true')
    ap.add_argument('--rerun-slam', action='store_true')
    ap.add_argument('--require-orb-slam3', action='store_true',
                    help='Require actual ORB-SLAM3 map factors; refuse a KLT-only reconstruction')
    ap.add_argument('-v', '--verbose', action='count', default=0)
    ap.add_argument('--data-root', type=Path, default=None,
                    help='recordings root, <id>/recording.tar (default: $UMI_DATA_DIR, else dataset)')
    ap.add_argument('--out', type=Path, required=True,
                    help='output root for per-episode outputs, calibration.json and summary.md '
                         '(e.g. dataset/reconstructions/<name>)')
    ap.add_argument('--calibration', type=Path, default=None,
                    help='camera/IMU calibration: a per-dataset calibration.json (camera_calibration --natural; '
                         'per-episode gyro bias priors) or a camera.json (default configs/camera.json); copied into --out')
    a = ap.parse_args()
    recording.set_data_root(a.data_root)
    pipeline.set_artifacts(a.out)
    import shutil
    dst = pipeline.ARTIFACTS / 'calibration.json'
    src = (a.calibration or camera.camera_path()).expanduser().resolve()
    if a.command == 'summary' and not a.calibration and dst.exists():
        src = dst.resolve()                  # describe the run with the calibration it used
    pipeline.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if src != dst.resolve():
        shutil.copyfile(src, dst)
        (pipeline.ARTIFACTS / 'calibration_source.txt').write_text(
            f'calibration.json copied from {src.relative_to(REPO) if src.is_relative_to(REPO) else src} (same camera, lens and camera-IMU mount; not re-estimated)\n')
    pipeline.set_artifacts(a.out, dst)
    eps = a.episodes or episode_ids()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.command in ('slam', 'all'):
        s = slam_all(a.work, eps, a.variant, rerun=a.rerun_slam)
        if a.command == 'all':
            other = [v for v in orbslam.VARIANTS if v != a.variant]
            for v in other:
                s.update({(k[0], f'{k[1]}-{v}'): x for k, x in slam_all(a.work, eps, v, modes=('mono',),
                                                                         rerun=a.rerun_slam).items()})
        (a.work / f'orbslam_{a.variant}.json').write_text(json.dumps({f'{k[0]}:{k[1]}': v for k, v in s.items()}, indent=1))
    if a.command in ('fuse', 'all'):
        failed = []
        for e in eps:
            try:
                process(e, a.work, a.variant, video=not a.no_video, sensitivity=not a.no_sensitivity,
                        verbose=a.verbose, require_orb_slam3=a.require_orb_slam3)
            except Exception as ex:   # keep going; record the failure for the summary
                import traceback
                traceback.print_exc()
                failed.append(e)
                out = pipeline.ARTIFACTS / e
                out.mkdir(parents=True, exist_ok=True)
                (out / 'summary.json').write_text(json.dumps({'episode': e, 'label': episode_label(e),
                    'failed': True, 'error': repr(ex), 'data_quality': {'manifest': recording.manifest_quality(e)}},
                    indent=2, default=str))
                print(f'{e}: FAILED {ex!r}', flush=True)
        if failed:
            print('failed episodes:', failed, flush=True)
            if a.require_orb_slam3:
                raise RuntimeError(f'Required ORB-SLAM3 processing failed for {len(failed)} episodes: {failed}')
    if a.command in ('summary', 'all'):
        from .summary import write_summary
        write_summary()


if __name__ == '__main__':
    main()
