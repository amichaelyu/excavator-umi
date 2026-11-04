"""policy.preprocessing: config-driven selection and split, example alignment, quality rules, export and freeze."""
import csv
import io
import json
import tarfile

import numpy as np
import pytest
import yaml

from policy import configs, preprocessing as pp
from policy.configs import ROOT
from policy.data.cloak_dataset import CloakDataset, validate_episode
from policy.training.cloak.policy import ActionOutputs, InferenceInputs

JOINTS = pp.JOINTS


def config(tmp_path, recordings, **source):
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw['source'] = {**raw['source'], 'recordings': recordings, **source}
    for section, kind in (('reconstruct', 'reconstructions'), ('labels', 'labels'), ('cohort', 'cohorts')):
        raw[section]['output'] = str(tmp_path / kind)
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(raw))
    return configs.load(path)


def recording(root, rid, label='dirt digging', start_ns=10**10, corrupt=(), duplicate=(), malformed=False, **meta):
    """<root>/<rid>/: metadata.json and a recording.tar of 50 small JPEG frames at 20 Hz (frame i coloured
    (200, 100 + i, 50)). ``corrupt`` frames hold garbage instead of a JPEG; ``duplicate`` frames.csv rows are
    written twice; ``malformed`` adds a truncated row."""
    from PIL import Image
    d = root / rid
    d.mkdir(parents=True)
    (d / 'metadata.json').write_text(json.dumps({'id': rid, 'label': label, 'error': '', 'dropped': 0,
                                                 'clipped_samples': 0, **meta}))
    frames, video = [], b''
    for i in range(50):
        buffer = io.BytesIO()
        Image.new('RGB', (64, 36), (200, 100 + i, 50)).save(buffer, 'JPEG')
        jpeg = b'\x00garbage' * 20 if i in corrupt else buffer.getvalue()
        row = {'camera_timestamp_ns': start_ns + i * 50_000_000, 'offset': len(video), 'size': len(jpeg)}
        frames += [row] * (2 if i in duplicate else 1)
        video += jpeg
    text = io.StringIO()
    writer = csv.DictWriter(text, ['camera_timestamp_ns', 'offset', 'size'])
    writer.writeheader(); writer.writerows(frames)
    if malformed:
        text.write(f'{start_ns + 50 * 50_000_000},12\n')
    with tarfile.open(d / 'recording.tar', 'w') as tar:
        for name, data in (('frames.csv', text.getvalue().encode()), ('video.mjpg', video)):
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return d


def labels_metrics(trajectory, **fields):
    """metrics.json fields that identify the replay's input, as digital_twin.sim.replay.run writes them: the
    trajectory digest from its loader, and ``input`` the path where the replay ran (here a Docker one that does not
    exist on this machine; export must never reopen it)."""
    from digital_twin.sim.replay import load_trajectory
    loader = load_trajectory(trajectory)[1]
    return {'input': f'/workspace/dataset/reconstructions/{trajectory.parent.name}/trajectory.csv',
            'input_sha256': loader['input_sha256'], 'loader': loader, **fields}


def reconstruction(vio_root, replay_root, rid, raw, start_ns=10**10):
    """A reconstructed trajectory and its IK replay (2 s of motion) for one recording."""
    vio, replay = vio_root / rid, replay_root / rid
    vio.mkdir(parents=True); replay.mkdir(parents=True)
    (vio / 'trajectory.csv').write_text('t_s,px,py,pz,qw,qx,qy,qz,status\n0,0,0,0,1,0,0,0,tracked\n'
                                        '2,0.1,0,0,1,0,0,0,tracked\n')
    (vio / 'summary.json').write_text(json.dumps({
        'result': 'initialized', 'warnings': [], 'quality': 'unvalidated', 'motion_source': 'orbslam3_smoother',
        'provenance': {'recording_sha256': pp.digest(raw)}, 'time_origin': {'raw_camera_timestamp_ns_at_t0': start_ns}}))
    t = np.arange(0, 2.01, .05)
    with (replay / 'actions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['t_s', 'source_t_s', *(f'q_{j}' for j in JOINTS)])
        writer.writeheader()
        writer.writerows({'t_s': x, 'source_t_s': x, **{f'q_{j}': x * (i + 1) for i, j in enumerate(JOINTS)}} for x in t)
    (replay / 'metrics.json').write_text(json.dumps(labels_metrics(
        vio / 'trajectory.csv', **{'planner': 'lookahead', 'profile': 'full4_calibrated',
        'ik': {'unreachable_count': 0}, 'retarget': {'pitch_feasible': True, 'scale': 1.0},
        'dynamic': {'unstable': False}, 'mount': {}, 'pitch': {'gain': 1.0},
        'timing': {'continuous_velocity_limit_ratio': .5, 'continuous_acceleration_limit_ratio': .5,
                   'cartesian_refinement': {'converged': True}}})))


def ids_with_both_splits(n, accepted):
    """n recording ids whose ``accepted`` slice (the ones the test lets through) holds both splits."""
    for k in range(1000):
        ids = [f'r{k}-{i}' for i in range(n)]
        if set(pp.holdout_split(ids)[i] for i in ids[accepted]) == {'train', 'holdout'}:
            return ids
    raise AssertionError('no id set found')


# --- split --------------------------------------------------------------------------------------------------------

def test_split_is_per_id_and_stable_as_recordings_are_added():
    first = pp.holdout_split([f'r{i}' for i in range(40)])
    later = pp.holdout_split([f'r{i}' for i in range(200)])
    assert all(later[i] == s for i, s in first.items())
    assert pp.holdout_split(list(reversed(list(first)))) == first
    assert 0.1 < sum(s == 'holdout' for s in later.values()) / len(later) < 0.3


def test_split_fills_an_empty_side_and_zero_means_all_train():
    assert set(pp.holdout_split(['a', 'b']).values()) == {'train', 'holdout'}
    assert set(pp.holdout_split([f'r{i}' for i in range(10)], 0).values()) == {'train'}
    assert pp.holdout_split(['only']) in ({'only': 'train'}, {'only': 'holdout'})
    with pytest.raises(ValueError, match='fraction'):
        pp.holdout_split(['a'], 1.0)


# --- select -------------------------------------------------------------------------------------------------------

def test_select_matches_labels_merges_sources_and_excludes(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for i in range(6):
        recording(a, f'a{i}', label='Dirt  Digging' if i % 2 else 'dirt digging')
    recording(a, 'cap', label='bottlecap')
    recording(b, 'b0', label='anything'); recording(b, 'b1', label='else')
    cfg = config(tmp_path, [{'data_root': str(a), 'label': 'dirt digging'},
                            {'data_root': str(b), 'ids': ['b1']}], exclude=['a5'])
    doc = pp.select(cfg)
    ids = [e['id'] for e in doc['episodes']]
    assert ids == ['a0', 'a1', 'a2', 'a3', 'a4', 'b1']
    assert {e['prompt'] for e in doc['episodes']} == {'scoop the dirt'} and doc['task_group'] == 'dirt'
    assert doc['excluded_by_config'] == ['a5'] and set(doc['split_counts']) == {'train', 'holdout'}
    assert doc['episodes'][0]['raw']['path'] == str(a / 'a0' / 'recording.tar')
    assert json.loads((tmp_path / 'cohorts' / 'selection.json').read_text()) == doc
    inputs = pp.reconstruct_inputs(cfg)
    assert [(str(root), ids) for root, _, ids in inputs] == [(str(a), ['a0', 'a1', 'a2', 'a3', 'a4']), (str(b), ['b1'])]
    assert json.loads((tmp_path / 'cohorts' / 'selection-1.json').read_text())['episodes'] == [{'id': 'b1'}]


def test_select_refuses_duplicates_unknown_excludes_and_wrong_ids(tmp_path):
    a = tmp_path / 'a'
    recording(a, 'x'); recording(a, 'y', label='other')
    with pytest.raises(ValueError, match='more than once'):
        pp.select(config(tmp_path, [{'data_root': str(a), 'label': 'dirt digging'}, {'data_root': str(a), 'ids': ['x']}]))
    with pytest.raises(ValueError, match='source.exclude'):
        pp.select(config(tmp_path, [{'data_root': str(a), 'label': 'dirt digging'}], exclude=['nope']))
    with pytest.raises(ValueError, match="is not 'dirt digging'"):
        pp.select(config(tmp_path, [{'data_root': str(a), 'label': 'dirt digging', 'ids': ['y']}]))
    with pytest.raises(FileNotFoundError, match='no data root'):
        pp.select(config(tmp_path, [{'data_root': str(tmp_path / 'missing'), 'label': 'x'}]))
    assert pp.select(config(tmp_path, [{'data_root': str(tmp_path / 'missing'), 'label': 'x'}]), dry_run=True)['episodes'] == []


def test_holdout_zero_selection_is_marked_none(tmp_path):
    a = tmp_path / 'a'
    for i in range(3):
        recording(a, f'r{i}')
    doc = pp.select(config(tmp_path, [{'data_root': str(a), 'label': 'dirt digging'}], holdout=0))
    assert doc['holdout'] == 'none' and doc['split_counts'] == {'train': 3}


# --- examples and quality -----------------------------------------------------------------------------------------

def test_retimed_actions_use_robot_clock_and_causal_camera_frames():
    robot = np.arange(0, 2.01, .05)
    # Strongly nonuniform motion. A source-clock interpolation of actions would be wrong.
    source = 5 + robot ** 2 / 4
    rows = [{'t_s': t, 'source_t_s': s, **{f'q_{j}': t * (i + 1) for i, j in enumerate(JOINTS)}}
            for t, s in zip(robot, source)]
    camera_ns = np.rint((10 + np.arange(0, 7.01, .05)) * 1e9).astype(np.int64)
    a = pp.aligned_examples(rows, camera_ns, 10000000000)
    assert len(a['state']) == 20
    assert (np.diff(a['source_frame']) == 0).any()  # Deliberate repeated RGB as the robot slows.
    assert np.all(a['source_time_s'] <= a['requested_source_time_s'] + 1e-7)
    np.testing.assert_allclose(a['actions'] - a['state'], np.tile([.1, .2, .3, .4], (20, 1)), atol=1e-6)
    assert a['time_s'][-1] + .1 <= robot[-1]
    a['images'] = np.zeros((20, 224, 224, 3), np.uint8)
    e = {'frames': 20, 'split': 'train', 'prompt': 'dig', 'action_contract': {
        'joints': JOINTS, 'type': 'absolute next joint target', 'units': 'radians',
        'sampling_hz': 10, 'time_base': 'retimed_robot_execution'}}
    validate_episode(e, a, masks=False)
    a['source_time_s'][2] = -1
    with pytest.raises(ValueError, match='source_time_s'):
        validate_episode(e, a, masks=False)


def test_four_joint_inference_keeps_base_and_stick():
    actions = np.arange(64).reshape(2, 32)
    np.testing.assert_array_equal(ActionOutputs(4)({'actions': actions})['actions'], actions[:, :4])
    obs = {'rgb': np.zeros((224, 224, 3), np.uint8), 'embodiment_mask': np.zeros((224, 224), bool),
           'state': [1, 2, 3, 4], 'prompt': 'dig'}
    np.testing.assert_array_equal(InferenceInputs('cloak_oov', 4)(obs)['state'], [1, 2, 3, 4])
    with pytest.raises(ValueError, match='requires 4'):
        InferenceInputs('cloak', 4)({**obs, 'state': [1, 2]})


def test_lenient_quality_includes_warning_trajectories_but_requires_usable_actions():
    summary = {'result': 'initialized', 'warnings': ['Motion spans over 3 m']}
    metrics = {'planner': 'lookahead', 'profile': 'full4_calibrated', 'ik': {'unreachable_count': 0},
               'retarget': {'pitch_feasible': True}, 'dynamic': {'unstable': False},
               'timing': {'continuous_velocity_limit_ratio': .98, 'continuous_acceleration_limit_ratio': .98,
                          'cartesian_refinement': {'converged': True}}}
    assert pp.quality_reasons({}, summary, metrics, 'lenient') == []
    assert pp.quality_reasons({}, summary, metrics, 'strict') == ['vio_warning_or_uninitialized']
    assert pp.quality_reasons({}, {**summary, 'result': 'not_initialized'}, metrics)
    assert pp.quality_reasons({'dropped': 2}, summary, metrics) == ['recorder_warning']
    assert pp.quality_reasons({'clipped_samples': 1}, summary, metrics) == ['recorder_warning']
    # Labels from the prior twin are only admitted when that twin is requested explicitly.
    prior = {**metrics, 'profile': 'full4'}
    assert pp.quality_reasons({}, summary, prior) == ['infeasible_or_incompatible_ik']
    assert pp.quality_reasons({}, summary, prior, profile='full4') == []
    assert pp.quality_reasons({}, summary, metrics, profile='full4') == ['infeasible_or_incompatible_ik']
    metrics['ik']['unreachable_count'] = 1
    assert pp.quality_reasons({}, summary, metrics) == ['infeasible_or_incompatible_ik']


# --- export and freeze --------------------------------------------------------------------------------------------

def test_export_and_freeze_from_a_config(tmp_path):
    data, vio, replay = tmp_path / 'data', tmp_path / 'reconstructions', tmp_path / 'labels'
    ids = ids_with_both_splits(6, slice(0, 4))
    for i, rid in enumerate(ids):
        d = recording(data, rid, start_ns=10**10 * (i + 1), **({'dropped': 3} if i == 5 else {}))
        if i != 4:                                             # the fifth was never reconstructed
            reconstruction(vio, replay, rid, d / 'recording.tar', 10**10 * (i + 1))
    cfg = config(tmp_path, [{'data_root': str(data), 'label': 'dirt digging'}])
    pp.cohort(cfg)
    prepared = json.loads((tmp_path / 'cohorts/export/prepared/manifest.json').read_text())
    accepted = {e['id']: e for e in prepared['episodes']}
    assert set(accepted) == set(ids[:4])
    assert {e['prompt'] for e in accepted.values()} == {'scoop the dirt'}
    assert prepared['source_embodiment'] == 'bucket1-handheld'
    assert all(e['raw']['sha256'] == pp.digest(data / i / 'recording.tar') for i, e in accepted.items())
    report = json.loads((tmp_path / 'cohorts/export/preparation.json').read_text())
    reasons = {e['id']: e['reasons'] for e in report['excluded']}
    assert reasons[ids[5]] == ['recorder_warning'] and ids[4] in reasons
    frozen = cfg.cohort_data
    split = pp.holdout_split(ids)
    train, held = CloakDataset(frozen), CloakDataset(frozen, 'holdout')
    assert {e['id'] for e, _ in train.episodes} == {i for i in accepted if split[i] == 'train'}
    assert {e['id'] for e, _ in held.episodes} == {i for i in accepted if split[i] == 'holdout'}
    cohort = json.loads((frozen.parent / 'cohort.json').read_text())
    assert cohort['membership'] == {i: split[i] for i in accepted}
    pp.cohort(cfg)                                             # rerun: every output exists, nothing changes


def test_export_refuses_a_recording_that_differs_from_its_reconstruction(tmp_path):
    data, vio, replay = tmp_path / 'data', tmp_path / 'reconstructions', tmp_path / 'labels'
    ids = ids_with_both_splits(4, slice(1, None))              # ids[0] changes; the rest still pair up
    for i, rid in enumerate(ids):
        d = recording(data, rid, start_ns=10**10 * (i + 1))
        reconstruction(vio, replay, rid, d / 'recording.tar', 10**10 * (i + 1))
    (data / ids[0] / 'recording.tar').write_bytes(b'changed')
    cfg = config(tmp_path, [{'data_root': str(data), 'label': 'dirt digging'}])
    pp.select(cfg)
    result = pp.export(tmp_path / 'cohorts/selection.json', tmp_path / 'out', vio, replay, visibility='out of view')
    assert result['episodes'] == 3 and result['excluded'] == 1
    report = json.loads((tmp_path / 'out/preparation.json').read_text())
    assert any('does not match the reconstruction' in r for e in report['excluded'] for r in e['reasons'])


def exported(tmp_path, n=4, accepted=slice(1, None), **recordings):
    """``n`` recordings reconstructed and labelled (``recordings``: {index: recording() options}), selected, and
    the export directory's arguments: (ids, data, vio, replay, export(output) -> result)."""
    data, vio, replay = tmp_path / 'data', tmp_path / 'reconstructions', tmp_path / 'labels'
    ids = ids_with_both_splits(n, accepted)
    for i, rid in enumerate(ids):
        d = recording(data, rid, start_ns=10**10 * (i + 1), **recordings.get(f'r{i}', {}))
        reconstruction(vio, replay, rid, d / 'recording.tar', 10**10 * (i + 1))
    pp.select(config(tmp_path, [{'data_root': str(data), 'label': 'dirt digging'}]))

    def run(output):
        return pp.export(tmp_path / 'cohorts/selection.json', output, vio, replay, visibility='out of view')
    return ids, data, vio, replay, run


def reasons_of(output):
    return {e['id']: e['reasons'] for e in json.loads((output / 'preparation.json').read_text())['excluded']}


def test_export_refuses_labels_of_an_older_trajectory_and_labels_redoes_them(tmp_path, monkeypatch):
    ids, data, vio, replay, run = exported(tmp_path, n=5, accepted=slice(2, None))
    # ids[0]: reconstructed again after its labels were made; ids[1]: labels from before input_sha256 existed.
    with (vio / ids[0] / 'trajectory.csv').open('a') as f:
        f.write('3,0.2,0,0,1,0,0,0,tracked\n')
    legacy = json.loads((replay / ids[1] / 'metrics.json').read_text())
    del legacy['input_sha256'], legacy['loader']['input_sha256']
    (replay / ids[1] / 'metrics.json').write_text(json.dumps(legacy))
    result = run(tmp_path / 'out')
    assert result['episodes'] == 3 and result['excluded'] == 2
    reasons = reasons_of(tmp_path / 'out')
    assert 'made from another trajectory.csv' in reasons[ids[0]][0]
    assert 'rerun the labels stage' in reasons[ids[1]][0]
    # The labels stage replays exactly those two again (and drops their stale metrics.json first).
    calls = []
    monkeypatch.setattr(pp, '_local', lambda cmd, dry_run=False, log=None: calls.append(cmd) or 0)
    pp.labels(config(tmp_path, [{'data_root': str(data), 'label': 'dirt digging'}]))
    replayed = sorted(c[c.index('--trajectory') + 1].parent.name for c in calls)
    assert replayed == sorted(ids[:2])
    assert not (replay / ids[0] / 'metrics.json').exists() and (replay / ids[2] / 'metrics.json').is_file()


def test_export_tolerates_damaged_frames_like_the_reconstruct_loader(tmp_path):
    # r1: a duplicate and a truncated frames.csv row and a corrupt sampled frame (4: frame 3 is 50 ms earlier);
    # r2: a corrupt sampled frame (6); r0: two in a row (3, 4: frame 2 is 100 ms before frame 4's sample), excluded.
    ids, data, vio, replay, run = exported(tmp_path, n=5, accepted=slice(1, None),
                                           r0={'corrupt': (3, 4)}, r1={'corrupt': (4,), 'duplicate': (10,), 'malformed': True},
                                           r2={'corrupt': (6,)})
    result = run(tmp_path / 'out')
    assert result['episodes'] == 4 and result['excluded'] == 1
    assert 'no earlier frame is within 75 ms' in reasons_of(tmp_path / 'out')[ids[0]][0]
    manifest = json.loads((tmp_path / 'out/source/manifest.json').read_text())
    entry = {e['id']: e for e in manifest['episodes']}
    assert entry[ids[1]]['undecodable_frames_replaced'] == [4] and entry[ids[2]]['undecodable_frames_replaced'] == [6]
    assert entry[ids[1]]['max_camera_lag_s'] == pytest.approx(.05)
    a = np.load(tmp_path / 'out/source' / entry[ids[1]]['path'])
    # source_frame counts frames.csv rows: frame 3 stands in for 4; after the duplicate row 10, frame k is row k + 1.
    assert a['source_frame'][:4].tolist() == [0, 2, 3, 6] and a['source_frame'][10] == 21
    centre = a['images'][:, 112, 112]
    assert abs(int(centre[2, 1]) - 103) <= 3 and abs(int(centre[3, 1]) - 106) <= 3     # frame 3 and frame 6


def test_one_failing_recording_never_aborts_the_export_and_a_crash_leaves_no_output(tmp_path, monkeypatch):
    ids, data, vio, replay, run = exported(tmp_path)
    with (replay / ids[0] / 'actions.csv').open('a') as f:
        f.write('2.05,2.05\n')                                   # a short row: float(None) is a TypeError
    real = pp.prepare_out_of_view
    monkeypatch.setattr(pp, 'prepare_out_of_view', lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        run(tmp_path / 'out')
    assert not (tmp_path / 'out').exists() and (tmp_path / '.out.partial/preparation.json').is_file()
    monkeypatch.setattr(pp, 'prepare_out_of_view', real)
    result = run(tmp_path / 'out')                               # rerun: the partial export is replaced
    assert result['episodes'] == 3 and not (tmp_path / '.out.partial').exists()
    assert reasons_of(tmp_path / 'out')[ids[0]][0].startswith('TypeError')
    with pytest.raises(FileExistsError):
        run(tmp_path / 'out')


@pytest.mark.parametrize('planner', ['ik', 'coupled'])
def test_cohort_refuses_labels_it_cannot_admit(tmp_path, planner):
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw['labels']['planner'] = planner
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(raw))
    cfg = configs.load(path)                                     # the labels stage may still replay with it
    with pytest.raises(ValueError, match='labels.planner: lookahead'):
        pp.cohort(cfg, dry_run=True)
    with pytest.raises(ValueError, match='labels.planner: lookahead'):
        pp.in_docker(cfg, 'cohort', dry_run=True)
