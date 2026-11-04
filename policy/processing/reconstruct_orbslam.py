"""Reconstruct reviewed real recordings with required ORB-SLAM3 map factors.

Outputs are a separate dataset candidate. Never overwrite the frozen pi0.5
training labels. A failed SLAM run is recorded and cannot silently use KLT alone.
The camera/IMU calibration (--calibration, default configs/camera.json; see policy.processing.slam.pipeline) is
copied into the output as calibration.json and used from there.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[2]


def run_one(args):
    eid, work, output, data, sensitivity = args
    from policy.processing.slam import __main__ as cli, pipeline, recording
    import cv2
    cv2.setNumThreads(1)
    pipeline.set_artifacts(output, output / 'calibration.json')
    recording.set_data_root(data)
    folder = output / eid
    folder.mkdir(exist_ok=True)
    with (folder / 'process.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            r = cli.process(eid, work, 'default', video=False, sensitivity=sensitivity,
                            require_orb_slam3=True)
            result = {'id': eid, 'status': 'reconstructed',
                      'plausible': r['plausibility_gate']['passed'],
                      'orb_factor_frames': r['slam_front_end']['frames_with_slam_segment_factors'],
                      'processed_frames': r['frames_processed'], 'sources': r['slam_front_end']['sources_in_fusion'],
                      'path_length_m': r['path_length_m'], 'max_speed_mps': r['max_speed_mps'],
                      'runtime_s': r['runtime_s']}
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'id': eid, 'status': 'failed', 'error': str(e)}
        (folder / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--calibration', type=Path, default=ROOT / 'configs' / 'camera.json',
                   help='camera.json, or a per-dataset calibration.json with per-episode gyro biases (default %(default)s)')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--episode')
    p.add_argument('--sensitivity', action='store_true')
    a = p.parse_args()
    for key in ('manifest', 'calibration', 'data_root', 'work', 'output'):
        setattr(a, key, getattr(a, key).resolve())
    if a.output == a.calibration.parent:
        raise ValueError('Use a separate output directory; preserve prior reconstruction')
    a.output.mkdir(parents=True, exist_ok=True); a.work.mkdir(parents=True, exist_ok=True)
    copy = a.output / 'calibration.json'
    if copy.exists() and copy.read_bytes() != a.calibration.read_bytes():
        raise ValueError('Output already uses a different calibration; choose a new output')
    shutil.copy2(a.calibration, copy)
    episodes = json.loads(a.manifest.read_text())['episodes']
    if a.episode:
        episodes = [e for e in episodes if e['id'] == a.episode]
    if not episodes:
        raise ValueError('No reviewed episodes')
    provenance = {'source_manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  'calibration_sha256': hashlib.sha256(copy.read_bytes()).hexdigest(),
                  'require_orb_slam3': True, 'sensitivity_evaluated': a.sensitivity,
                  'calibration_source': str(a.calibration.relative_to(ROOT) if a.calibration.is_relative_to(ROOT) else a.calibration),
                  'calibration_note': 'Existing camera/IMU rotation and time offset; unmeasured lever arm assumed zero. Metric accuracy is not established by tracking coverage.',
                  'source': 'real recordings only', 'episodes': []}
    start = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(run_one, (e['id'], a.work, a.output, a.data_root, a.sensitivity)) for e in episodes]
        for f in as_completed(jobs):
            row = f.result(); provenance['episodes'].append(row)
            provenance['runtime_s'] = time.time() - start
            (a.output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f'{len(provenance["episodes"])}/{len(jobs)} {json.dumps(row)}', flush=True)
    failed = sum(r['status'] == 'failed' for r in provenance['episodes'])
    print(f'Completed {len(episodes)} recordings; {failed} failed explicitly.', flush=True)


if __name__ == '__main__':
    main()
