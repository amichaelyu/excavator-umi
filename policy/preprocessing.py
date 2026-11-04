"""Preprocessing, driven by the config: recordings -> four-joint action labels -> a frozen train/holdout cohort.

    select   the recordings ``source.recordings`` names (a data root each, picked by label or by ids), minus
             ``source.exclude``, each held out or not by a hash of its id        -> <cohort>/selection.json
    labels   twin IK replay of every reconstructed trajectory                     -> <labels>/<id>/actions.csv
    cohort   quality filter, camera-aligned examples, embodiment masks, frozen dataset -> cfg.cohort_data

``python -m policy labels|cohort <config>`` runs a stage; the reconstruct stage (policy.training.stages) selects
first. Every stage skips work whose output exists, so it can be rerun after a failure. ``<stage>.runner: docker``
runs it in the umi-excavator-preprocess image (policy/docker/preprocess.Dockerfile).

A recording is ``<data_root>/<id>/``: ``metadata.json`` (label, recorder error, frame and IMU counts) and
``recording.tar`` (frames.csv, imu.csv, video.mjpg). The task and its prompt come from the config's ``task``; the
labels are reconstructed (pose -> IK), not measured. The frozen dataset format is policy.data.cloak_dataset,
which training reads. This module imports only configs, data, sim and processing (policy/tests/test_layering.py).
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from policy.configs import ROOT
from policy.configs.schema import LABEL_PROFILES, POLICY_HZ, TASKS
from policy.data.cloak_dataset import CloakDataset, digest, local_path, prepare_out_of_view, validate_episode

DOCKERFILE = ROOT / 'policy' / 'docker' / 'preprocess.Dockerfile'
SELECTION_SCHEMA = 'actathon.selection.v2'
JOINTS = ['slew', 'boom', 'stick', 'bucket']
DEFAULT_PROFILE = 'full4_calibrated'
DEFAULT_HOLDOUT = 0.2
DEFAULT_EMBODIMENT = 'bucket1-handheld'
SPLIT_SALT = 'split-v1/'
SPLIT_RULE = ('Whole recording: held out when sha256("split-v1/" + id) falls below the holdout fraction, so a '
              'recording keeps its split as others are added; at least one of each split when there are two.')
MOTION_SOURCES = {'orbslam3_smoother': 'ORB-SLAM3 map factors plus KLT two-view and IMU batch smoothing'}
IMAGE_SIZE = (224, 224)          # letterboxed RGB; the 1280x720 frame fills rows CONTENT_BOX[1]:CONTENT_BOX[3]
CONTENT_BOX = [0, 49, 224, 175]
MAX_CAMERA_LAG_S = .075


# ------------------------------------------------------------------------------------------------- running
def _local(cmd, dry_run=False, log=None):
    cmd = [str(c) for c in cmd]
    print('+ ' + shlex.join(cmd), file=sys.stderr, flush=True)
    if dry_run:
        return 0
    if log is None:
        return subprocess.run(cmd, cwd=ROOT, check=True).returncode
    with open(log, 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def image_tag():
    """umi-excavator-preprocess tagged by a hash of what it installs."""
    h = hashlib.sha256()
    for path in (DOCKERFILE, ROOT / 'pyproject.toml', ROOT / 'uv.lock', ROOT / '.python-version'):
        h.update(path.read_bytes())
    return f'umi-excavator-preprocess:{h.hexdigest()[:12]}'


def in_docker(cfg, stage, dry_run=False):
    """Run ``python -m policy <stage> <config> --runner local`` in the preprocessing image (built here when this
    tag is missing), the repository at /workspace. A config outside the repository is mounted read-only."""
    tag = image_tag()
    mounts = ['-v', f'{ROOT}:/workspace']
    config = cfg.path
    if config.is_relative_to(ROOT):
        config = config.relative_to(ROOT)
    else:
        mounts += ['-v', f'{config}:{config}:ro']
    build = ['docker', 'build', '-f', DOCKERFILE, '-t', tag, '-t', 'umi-excavator-preprocess:latest', ROOT]
    run = ['docker', 'run', '--rm', '--init', '--user', f'{os.getuid()}:{os.getgid()}', *mounts, '-w', '/workspace',
           tag, 'python', '-m', 'policy', stage, config, '--runner', 'local']
    if dry_run or subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True).returncode:
        _local(build, dry_run)
    _local(run, dry_run)


def _repo_path(path):
    """A config path: absolute as given, else relative to the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


# ------------------------------------------------------------------------------------------------- select
def _label(text):
    return ' '.join(str(text or '').lower().split())


def holdout_split(ids, fraction=DEFAULT_HOLDOUT):
    """{id: 'train' | 'holdout'}: held out when sha256(SPLIT_SALT + id) / 2**256 < ``fraction`` (0: all train).
    Each id's split depends only on the id, so adding recordings never moves the existing ones; with two or more
    recordings and a nonzero fraction, the id nearest the threshold fills a split that would otherwise be empty."""
    if not 0 <= fraction < 1:
        raise ValueError('source.holdout must be a fraction in [0, 1)')
    keys = {i: int(hashlib.sha256((SPLIT_SALT + i).encode()).hexdigest()[:8], 16) / 2**32 for i in ids}
    result = {i: 'holdout' if k < fraction else 'train' for i, k in keys.items()}
    if fraction and len(result) >= 2:
        if 'holdout' not in result.values():
            result[min(keys, key=keys.get)] = 'holdout'
        if 'train' not in result.values():
            result[max(keys, key=keys.get)] = 'train'
    return result


def _recordings(spec, task, dry_run):
    """Selection rows for one ``source.recordings`` entry (``data_root`` plus ``label`` and / or ``ids``)."""
    root = Path(spec['data_root'])
    base = _repo_path(root)
    if not base.is_dir():
        if dry_run:
            print(f'# {root}: no recordings here yet', file=sys.stderr)
            return []
        raise FileNotFoundError(f'source.recordings: no data root {root}')
    ids = spec.get('ids')
    if ids is None:
        ids = sorted(d.name for d in base.iterdir() if (d / 'metadata.json').is_file())
    rows = []
    for rid in ids:
        path = base / rid / 'metadata.json'
        if not path.is_file():
            raise FileNotFoundError(f'{root}/{rid}: no metadata.json')
        meta = json.loads(path.read_text())
        if spec.get('label') is not None and _label(meta.get('label')) != _label(spec['label']):
            if spec.get('ids') is not None:
                raise ValueError(f'{root}/{rid}: label {meta.get("label")!r} is not {spec["label"]!r}')
            continue
        rows.append({'id': rid, 'data_root': str(root), 'label': meta.get('label') or '', 'group': task,
                     'prompt': TASKS[task], 'raw': {'path': str(root / rid / 'recording.tar')},
                     'video': {'path': str(root / rid / 'video.mp4')},
                     'recorder_error': meta.get('error') or '', 'dropped': meta.get('dropped', 0),
                     'clipped_samples': meta.get('clipped_samples', 0)})
    return rows


def select(cfg, dry_run=False):
    """The selection document (written to ``<cohort>/selection.json`` unless ``dry_run``): every recording the
    config's ``source.recordings`` names, minus ``source.exclude``, each with its train / holdout split."""
    src = cfg.section('source')
    rows = [row for spec in src['recordings'] for row in _recordings(spec, cfg.task, dry_run)]
    seen = Counter(r['id'] for r in rows)
    if any(n > 1 for n in seen.values()):
        raise ValueError(f'Recordings selected more than once: {sorted(i for i, n in seen.items() if n > 1)}')
    exclude = set(src.get('exclude', []))
    if exclude - set(seen) and not (dry_run and not rows):
        raise ValueError(f'source.exclude names recordings not selected: {sorted(exclude - set(seen))}')
    rows = [r for r in rows if r['id'] not in exclude]
    fraction = src.get('holdout', DEFAULT_HOLDOUT)
    split = holdout_split([r['id'] for r in rows], fraction)
    rows = sorted(({**r, 'split': split[r['id']]} for r in rows), key=lambda r: r['id'])
    document = {'schema': SELECTION_SCHEMA, 'task_group': cfg.task, 'prompt': TASKS[cfg.task],
                **({'holdout': 'none'} if fraction == 0 else {}), 'holdout_fraction': fraction,
                'split_rule': SPLIT_RULE, 'sources': src['recordings'], 'excluded_by_config': sorted(exclude),
                'episode_count': len(rows), 'split_counts': dict(Counter(r['split'] for r in rows)),
                'episodes': rows}
    if not dry_run:
        out = ROOT / cfg.dataset('cohorts') / 'selection.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        temporary.replace(out)
    return document


def reconstruct_inputs(cfg, dry_run=False):
    """[(data_root, manifest, ids)] for the reconstruct stage: one per data root, its selected recordings listed in
    ``manifest`` (``<cohort>/selection.json`` itself when there is one data root)."""
    document = select(cfg, dry_run)
    roots = list(dict.fromkeys(str(Path(spec['data_root'])) for spec in cfg.section('source')['recordings']))
    base = cfg.dataset('cohorts')
    result = []
    for i, root in enumerate(roots):
        ids = [e['id'] for e in document['episodes'] if e['data_root'] == root]
        if len(roots) == 1:
            manifest = base / 'selection.json'
        else:
            manifest = base / f'selection-{i}.json'
            if not dry_run:
                (ROOT / manifest).write_text(json.dumps({'data_root': root, 'episodes': [{'id': x} for x in ids]},
                                                        indent=2) + '\n')
        if ids or dry_run:
            result.append((Path(root), manifest, ids))
    return result


# ------------------------------------------------------------------------------------------------- labels
def labels(cfg, dry_run=False):
    """IK replay of each trajectory in the twin -> four-joint action labels (``<output>/<id>/actions.csv``)."""
    lab = cfg.section('labels')
    vio, out = ROOT / cfg.dataset('reconstructions'), ROOT / cfg.dataset('labels')
    trajectories = sorted(vio.glob('*/trajectory.csv'))
    if not trajectories and not dry_run:
        raise FileNotFoundError(f'No trajectories under {vio}; run the reconstruct stage first')
    options = ['--profile', lab.get('profile', DEFAULT_PROFILE), '--planner', lab.get('planner', 'lookahead'),
               '--no-render']
    if lab.get('contact_height') is not None:
        options += ['--contact-height', lab['contact_height']]
    if lab.get('pitch_clip'):
        options.append('--pitch-clip')

    def replay(trajectory):
        target = out / trajectory.parent.name
        if (target / 'metrics.json').is_file():
            return 0
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
        return _local([sys.executable, '-m', 'digital_twin.sim.replay', '--trajectory', trajectory,
                       '--output', target, *options], dry_run, None if dry_run else target / 'replay.log')

    if dry_run and not trajectories:
        print(f'# no trajectories under {cfg.dataset("reconstructions")} yet; for each <id> the reconstruct stage writes:',
              file=sys.stderr)
        _local([sys.executable, '-m', 'digital_twin.sim.replay', '--trajectory',
                cfg.dataset('reconstructions') / '<id>' / 'trajectory.csv', '--output', cfg.dataset('labels') / '<id>',
                *options], dry_run)
    with ThreadPoolExecutor(lab.get('workers', 8)) as pool:
        failed = [t.parent.name for t, rc in zip(trajectories, pool.map(replay, trajectories)) if rc]
    if failed:
        print(f'{len(failed)} replays failed (see <labels>/<id>/replay.log): {failed}', file=sys.stderr)


# ------------------------------------------------------------------------------------------------- export
def aligned_examples(action_rows, camera_ns, origin_ns, hz=POLICY_HZ):
    """Use source_t_s for camera lookup, t_s for future joint targets. No future RGB."""
    robot = np.asarray([float(r['t_s']) for r in action_rows])
    source = np.asarray([float(r['source_t_s']) for r in action_rows])
    joints = np.asarray([[float(r[f'q_{j}']) for j in JOINTS] for r in action_rows])
    camera_ns = np.asarray(camera_ns, np.int64)
    if (hz <= 0 or len(robot) < 3 or not np.isfinite([robot, source]).all()
            or not np.isfinite(joints).all() or np.any(np.diff(robot) <= 0)
            or np.any(np.diff(source) <= 0) or np.any(np.diff(camera_ns) <= 0)):
        raise ValueError('Invalid source, camera, or execution clock')
    # Every observation needs one actual future command, never an extrapolated target.
    time = robot[0] + np.arange(int(np.floor((robot[-1] - robot[0]) * hz + 1e-6))) / hz
    mapped = np.interp(time, robot, source)
    camera = (camera_ns - int(origin_ns)) * 1e-9
    frame = np.searchsorted(camera, mapped + 1e-7, side='right') - 1
    if len(time) < 10 or (frame < 0).any():
        raise ValueError('Too little initialized motion with an available camera frame')
    lag = mapped - camera[frame]
    if (lag > MAX_CAMERA_LAG_S).any() or (lag < -1e-6).any():
        raise ValueError('Camera clock mismatch or a gap over 75 ms')
    state = np.column_stack([np.interp(time, robot, joints[:, j]) for j in range(len(JOINTS))])
    actions = np.column_stack([np.interp(time + 1 / hz, robot, joints[:, j]) for j in range(len(JOINTS))])
    return {'state': state.astype(np.float32), 'actions': actions.astype(np.float32),
            'source_frame': frame, 'time_s': time, 'source_time_s': camera[frame],
            'requested_source_time_s': mapped, 'camera_lag_s': lag}


def quality_reasons(entry, summary, metrics, policy='lenient', profile=DEFAULT_PROFILE, max_clip_fraction=None,
                    max_clip_deg=None):
    """Exclusion reasons. A replay made with --pitch-clip may have no single feasible pitch offset; its
    saturated samples are admitted unless they exceed max_clip_fraction / max_clip_deg (None = accept)."""
    if policy not in ('strict', 'lenient'):
        raise ValueError('Unknown motion quality policy')
    if profile not in LABEL_PROFILES:
        raise ValueError('Unknown label twin profile')
    reasons = []
    if entry.get('recorder_error') or entry.get('dropped') or entry.get('clipped_samples'):
        reasons.append('recorder_warning')
    if summary.get('result') != 'initialized' or (policy == 'strict' and summary.get('warnings')):
        reasons.append('vio_warning_or_uninitialized')
    if metrics is None:
        reasons.append('missing_ik')
    elif (metrics.get('planner') != 'lookahead' or metrics.get('profile') != profile
            or metrics['ik']['unreachable_count']
            or not (metrics['retarget']['pitch_feasible'] or metrics.get('pitch_clip', {}).get('enabled'))
            or metrics['dynamic']['unstable']):
        reasons.append('infeasible_or_incompatible_ik')
    elif (metrics['timing']['continuous_velocity_limit_ratio'] > 1
            or metrics['timing']['continuous_acceleration_limit_ratio'] > 1
            or not metrics['timing']['cartesian_refinement']['converged']):
        reasons.append('ik_limits_or_interpolation')
    clip = (metrics or {}).get('pitch_clip') or {}
    if ((max_clip_fraction is not None and clip.get('fraction', 0) > max_clip_fraction)
            or (max_clip_deg is not None and clip.get('max_deg', 0) > max_clip_deg)):
        reasons.append('pitch_clip_over_threshold')
    return reasons


def _letterbox(jpeg):
    from PIL import Image, ImageOps
    image = Image.open(io.BytesIO(jpeg)).convert('RGB')
    return np.asarray(ImageOps.pad(image, IMAGE_SIZE, method=Image.Resampling.BILINEAR, color=0))


def export(selection, output, vio_root, replay_root, *, visibility, embodiment=DEFAULT_EMBODIMENT,
           quality_policy='lenient', motion_source='orbslam3_smoother', profile=DEFAULT_PROFILE,
           max_clip_fraction=None, max_clip_deg=None):
    """Quality-filtered, camera-aligned examples of every selected recording (``output/source``: one npz each, an
    actathon.pi05.real.v1 manifest, ``output/preparation.json``), then the out-of-view embodiment dataset
    (``output/prepared``: empty masks with the ``visibility`` confirmation)."""
    selection, output, vio_root, replay_root = Path(selection), Path(output), Path(vio_root), Path(replay_root)
    plan = json.loads(selection.read_text())
    if plan.get('schema') != SELECTION_SCHEMA:
        raise ValueError(f'Expected a {SELECTION_SCHEMA} selection (policy.preprocessing.select)')
    if not visibility or not str(visibility).strip():
        raise ValueError('source.visibility (the source tool is out of view) is required')
    if motion_source not in MOTION_SOURCES:
        raise ValueError('Unknown motion source')
    if profile not in LABEL_PROFILES:
        raise ValueError('Unknown label twin profile')
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'; source.mkdir()
    accepted, excluded = [], []
    contract = {'joints': JOINTS, 'units': 'radians', 'type': 'absolute next joint target',
                'sampling_hz': POLICY_HZ, 'time_base': 'retimed_robot_execution', 'profile': profile,
                'policy_space': LABEL_PROFILES[profile],
                'provenance': f'{MOTION_SOURCES[motion_source]} pose, four-joint lookahead IK in the {profile} twin, '
                              'reconstructed commands and states; no measured joint labels'}
    for entry in plan['episodes']:
        eid = entry['id']
        if entry.get('group') != plan['task_group']:
            raise ValueError('Cross-task contamination in selection')
        try:
            sp = vio_root / eid / 'summary.json'
            summary = json.loads(sp.read_text())
            if summary.get('motion_source') != motion_source:
                raise ValueError('Motion summary belongs to another estimator')
            if summary.get('result') != 'initialized':
                excluded.append({'id': eid, 'split': entry['split'], 'reasons': ['no_initialized_pose'],
                                 'vio_warnings': summary.get('warnings', [])})
                continue
            mp = replay_root / eid / 'metrics.json'
            metrics = json.loads(mp.read_text())
            reasons = quality_reasons(entry, summary, metrics, quality_policy, profile, max_clip_fraction, max_clip_deg)
            if reasons:
                excluded.append({'id': eid, 'split': entry['split'], 'reasons': reasons,
                                 'vio_warnings': summary.get('warnings', [])})
                continue
            raw = _repo_path(entry['raw']['path'])
            raw_sha = digest(raw)
            if summary['provenance']['recording_sha256'] != raw_sha:
                raise ValueError('Raw recording does not match the reconstruction')
            if digest(_repo_path(metrics['input'])) != digest(sp.parent / 'trajectory.csv'):
                raise ValueError('IK replay and motion trajectory differ')
            with tarfile.open(raw) as tar:
                frames = list(csv.DictReader(io.TextIOWrapper(tar.extractfile('frames.csv'))))
                video = tar.extractfile('video.mjpg').read()
            with (mp.parent / 'actions.csv').open() as f:
                actions = list(csv.DictReader(f))
            # These change what the labels mean: one value per dataset (contract identity).
            gain = float(metrics.get('pitch', {}).get('gain', 1.0))
            if contract.setdefault('bucket_pitch_gain', gain) != gain:
                raise ValueError('IK replays use different bucket pitch gains')
            clip = metrics.get('pitch_clip') or {}
            if contract.setdefault('bucket_pitch_clip', bool(clip.get('enabled'))) != bool(clip.get('enabled')):
                raise ValueError('IK replays mix clipped and unclipped bucket pitch')
            # --contact-height: the angle is enforced only in the dirt (digital_twin.sim.replay).
            height = (metrics.get('contact') or {}).get('height_m')
            if contract.setdefault('bucket_pitch_contact_height_m', height) != height:
                raise ValueError('IK replays use different contact heights')
            a = aligned_examples(actions, [int(f['camera_timestamp_ns']) for f in frames],
                                 summary['time_origin']['raw_camera_timestamp_ns_at_t0'])
            images = {}
            for i in np.unique(a['source_frame']):
                offset, size = int(frames[i]['offset']), int(frames[i]['size'])
                images[i] = _letterbox(video[offset:offset + size])
            a['images'] = np.stack([images[i] for i in a['source_frame']])
            record = {**entry, 'raw': {**entry['raw'], 'sha256': raw_sha}, 'frames': len(a['state']),
                      'action_contract': contract, 'motion_source': motion_source,
                      'quality': 'unvalidated, admitted by the declared motion quality policy',
                      'motion_quality_policy': quality_policy, 'motion_warnings': summary.get('warnings', []),
                      'source_quality': summary['quality'], 'mount_measured': metrics['mount'].get('measured', False),
                      'retarget_scale': metrics['retarget']['scale'],
                      'pitch_clip': {k: clip.get(k, 0.0) for k in ('fraction', 'max_deg', 'mean_deg_clipped')},
                      'trajectory_sha256': digest(sp.parent / 'trajectory.csv'),
                      'motion_summary_sha256': digest(sp), 'ik_metrics_sha256': digest(mp),
                      'actions_csv_sha256': digest(mp.parent / 'actions.csv'),
                      'camera_alignment': 'Latest captured RGB at or before source_t_s; repeated images allowed as the robot slows',
                      'max_camera_lag_s': float(a['camera_lag_s'].max()), 'path': f'{eid}.npz'}
            validate_episode(record, a, masks=False)
            np.savez_compressed(source / record['path'], **a)
            record['sha256'] = digest(source / record['path'])
            accepted.append(record)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            excluded.append({'id': eid, 'split': entry['split'], 'reasons': [str(exc)]})
    clipped = [e for e in accepted if e['pitch_clip']['fraction'] > 0]
    report = {'task_group': plan['task_group'], 'source_selection_sha256': digest(selection),
              'label_profile': profile, 'policy_space': LABEL_PROFILES[profile],
              'pitch_clip': {'max_fraction': max_clip_fraction, 'max_deg': max_clip_deg,
                             'accepted_with_clipping': len(clipped),
                             'per_recording': {e['id']: e['pitch_clip'] for e in clipped}},
              'accepted': [{'id': e['id'], 'split': e['split'], 'frames': e['frames']} for e in accepted],
              'excluded': excluded, 'motion_quality_policy': quality_policy,
              'accepted_with_motion_warnings': sum(bool(e['motion_warnings']) for e in accepted),
              'quality_rule': 'Lenient includes motion warnings at full weight. Strict excludes them. Both exclude '
                              'missing poses, recorder faults, infeasible IK, limit violations, and invalid '
                              'alignment/provenance. Labels remain unvalidated.'}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    none = plan.get('holdout') == 'none'
    if {e['split'] for e in accepted} != ({'train'} if none else {'train', 'holdout'}):
        raise ValueError('holdout: none selections must mark every recording train' if none else
                         'Quality filtering left no train/holdout pair; see preparation.json')
    manifest = {'schema': 'actathon.pi05.real.v1', 'task_group': plan['task_group'],
                'motion_source': motion_source, 'label_profile': profile,
                'pitch_clip_thresholds': {'max_fraction': max_clip_fraction, 'max_deg': max_clip_deg},
                'motion_quality_policy': quality_policy, **({'holdout': 'none'} if none else {}),
                'episodes': accepted, 'source_selection_sha256': digest(selection),
                'limitations': ['Joint labels and metric scale are reconstructed and unvalidated.',
                                'Camera-to-bucket mount is unmeasured.',
                                'Task success and physical deployment have not been validated.'],
                'excluded': excluded, 'split_rule': plan['split_rule']}
    (source / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    result = prepare_out_of_view(source, output / 'prepared', embodiment=embodiment,
                                 evidence={'confirmation': str(visibility).strip(),
                                           'source_manifest_sha256': digest(source / 'manifest.json')},
                                 content_box=CONTENT_BOX)
    return {'task_group': plan['task_group'], 'episodes': len(result['episodes']),
            'splits': dict(Counter(e['split'] for e in accepted)), 'excluded': len(excluded),
            'frames': sum(e['frames'] for e in accepted)}


# ------------------------------------------------------------------------------------------------- freeze
def freeze(method, prepared, output):
    """Freeze the prepared dataset under ``output/<method>`` (hard links) with ``output/cohort.json``."""
    prepared, output = Path(prepared), Path(output)
    doc = json.loads((prepared / 'manifest.json').read_text())
    entries = doc['episodes']
    if not entries:
        raise ValueError('No accepted recordings to freeze')
    for entry in entries:
        if entry['action_contract']['joints'] != JOINTS:
            raise ValueError('Four-joint targets are required')
    if len({json.dumps({k: e['action_contract'].get(k) for k in ('profile', 'policy_space')}) for e in entries}) != 1:
        raise ValueError('Labels from different twins')
    membership = {e['id']: e['split'] for e in entries}
    none = doc.get('holdout') == 'none'
    if set(membership.values()) != ({'train'} if none else {'train', 'holdout'}):
        raise ValueError('The cohort needs train and holdout recordings (or holdout: none, all train)')
    dest = output / method
    dest.mkdir(parents=True, exist_ok=False)
    frozen = copy.deepcopy(doc)
    frozen['frozen_from_manifest_sha256'] = digest(prepared / 'manifest.json')
    for entry in frozen['episodes']:
        source = local_path(prepared, entry['path'])
        if digest(source) != entry['sha256']:
            raise ValueError('Source episode checksum mismatch')
        target = local_path(dest, entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        try:                                  # hard links keep large immutable arrays inside each dataset root
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    (dest / 'manifest.json').write_text(json.dumps(frozen, indent=2) + '\n')
    train, held = CloakDataset(dest), CloakDataset(dest, 'holdout')
    contract = entries[0]['action_contract']
    report = {'schema': 'actathon.cohort.v2', 'task_group': doc['task_group'], 'membership': membership,
              'label_profile': contract.get('profile'), 'policy_space': contract.get('policy_space'),
              'methods': {method: {'manifest_sha256': digest(dest / 'manifest.json'),
                                   'train_recordings': len(train.episodes), 'holdout_recordings': len(held.episodes),
                                   'train_frames': len(train), 'holdout_frames': len(held)}},
              'scope': f'One {method} cohort; every mode trains on identical examples. '
                       'Labels are not independent ground truth.'}
    (output / 'cohort.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


# ------------------------------------------------------------------------------------------------- cohort
def cohort(cfg, dry_run=False):
    """Export labelled examples and freeze the train/holdout cohort the run trains on.

    ``export`` writes ``<cohort>/export/source`` (the examples) and ``export/prepared`` (empty embodiment masks
    with the out-of-view confirmation: baseline, cloak_oov). ``cohort.masks: twin`` (default for cloak / noaug)
    renders twin self-masks at each labelled state into ``export/twin-masks`` and joins them into
    ``export/prepared-twin``. The chosen dataset is frozen into ``cfg.cohort_data``.
    """
    src, coh, lab = cfg.section('source'), cfg.section('cohort'), cfg.section('labels')
    base = cfg.dataset('cohorts')
    exported = base / 'export'
    if not (ROOT / exported / 'prepared' / 'manifest.json').is_file():
        print(f'# export {base / "selection.json"} -> {exported} ({cfg.method} motion, '
              f'quality {coh.get("quality_policy", "lenient")})', file=sys.stderr, flush=True)
        if not dry_run:
            select(cfg)
            result = export(ROOT / base / 'selection.json', ROOT / exported, ROOT / cfg.dataset('reconstructions'),
                            ROOT / cfg.dataset('labels'), visibility=src['visibility'],
                            embodiment=src.get('embodiment', DEFAULT_EMBODIMENT),
                            quality_policy=coh.get('quality_policy', 'lenient'), motion_source=cfg.method,
                            profile=lab.get('profile', DEFAULT_PROFILE),
                            max_clip_fraction=coh.get('max_pitch_clip_fraction'),
                            max_clip_deg=coh.get('max_pitch_clip_deg'))
            print(json.dumps(result, indent=2))
    prepared = exported / 'prepared'
    if cfg.masks == 'twin':
        masks, prepared = exported / 'twin-masks', exported / 'prepared-twin'
        if not (ROOT / masks / 'manifest.json').is_file():
            _local([sys.executable, '-m', 'policy.training.cloak.twin_masks', '--data', exported / 'source',
                    '--output', masks, *(['--config', coh['mask_config']] if coh.get('mask_config') else []),
                    '--workers', coh.get('mask_workers', 8)], dry_run)
        if not (ROOT / prepared / 'manifest.json').is_file():
            _local([sys.executable, '-m', 'policy.training.cloak', 'prepare', '--data', exported / 'source',
                    '--masks', masks, '--output', prepared], dry_run)
    frozen = cfg.cohort_data.parent
    if (ROOT / frozen).exists():
        return
    print(f'# freeze {prepared} -> {cfg.cohort_data}', file=sys.stderr)
    if dry_run:
        return
    report = freeze(cfg.method, ROOT / prepared, ROOT / frozen)
    print(json.dumps(report['methods'], indent=2))
