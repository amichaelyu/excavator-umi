"""Runner selection: native binaries inside the umi-excavator-processing image (env vars), Docker fallback elsewhere."""
import pytest

from policy.processing.slam import orbslam


@pytest.fixture
def no_image_env(monkeypatch):
    for name in ('ORBSLAM3_PROBE', 'ORBSLAM3_VOCABULARY'):
        monkeypatch.delenv(name, raising=False)


def paths(tmp_path):
    export, outdir = tmp_path / 'exports/raw/ep', tmp_path / 'results/default-imu/ep'
    return tmp_path, export, tmp_path / 'settings/ep/default-imu.yaml', outdir


def test_orbslam_falls_back_to_docker_without_env(tmp_path, no_image_env):
    assert orbslam.native_probe() is None
    data, export, settings, outdir = paths(tmp_path)
    cmd, env = orbslam.command(data, export, settings, outdir, 'imu', timeout_s=60)
    assert cmd[:3] == ['bash', str(orbslam.RUN_SH), 'processing']
    assert cmd[3:7] == ['timeout', '--signal=INT', '--kill-after=10', '60s']
    assert cmd[7:9] == [orbslam.IMAGE_PROBE, orbslam.IMAGE_VOCABULARY]
    assert cmd[9:] == ['/data/settings/ep/default-imu.yaml', '/data/exports/raw/ep',
                       '/data/exports/raw/ep/camera_timestamps.txt', '/data/results/default-imu/ep', 'imu']
    assert env == {'UMI_DATA_DIR': str(tmp_path.resolve())}


def test_orbslam_docker_fallback_requires_paths_under_data_dir(tmp_path, no_image_env):
    data, export, settings, outdir = paths(tmp_path / 'data')
    with pytest.raises(ValueError):
        orbslam.command(data, tmp_path / 'elsewhere', settings, outdir, 'mono')


def test_orbslam_runs_native_probe_from_env(tmp_path, monkeypatch):
    probe, vocabulary = tmp_path / 'orbslam_probe', tmp_path / 'ORBvoc.txt'
    probe.write_text(''); vocabulary.write_text('')
    monkeypatch.setenv('ORBSLAM3_PROBE', str(probe))
    monkeypatch.setenv('ORBSLAM3_VOCABULARY', str(vocabulary))
    assert orbslam.native_probe() == (probe, vocabulary)
    data, export, settings, outdir = paths(tmp_path)
    # Native paths need not live under the data directory.
    other = tmp_path.parent / 'elsewhere'
    cmd, env = orbslam.command(data, export, settings, other, 'mono', timeout_s=5)
    assert env is None
    assert cmd == ['timeout', '--signal=INT', '--kill-after=10', '5s', str(probe), str(vocabulary),
                   str(settings.resolve()), str(export.resolve()), str((export / 'camera_timestamps.txt').resolve()),
                   str(other.resolve()), 'mono']


def test_orbslam_native_vocabulary_defaults_to_image_path(tmp_path, monkeypatch):
    probe = tmp_path / 'orbslam_probe'
    probe.write_text('')
    monkeypatch.setenv('ORBSLAM3_PROBE', str(probe))
    monkeypatch.delenv('ORBSLAM3_VOCABULARY', raising=False)
    monkeypatch.setattr(orbslam, 'IMAGE_VOCABULARY', str(tmp_path / 'missing.txt'))
    with pytest.raises(FileNotFoundError, match='missing.txt'):
        orbslam.native_probe()


def test_orbslam_missing_native_probe_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv('ORBSLAM3_PROBE', str(tmp_path / 'absent'))
    with pytest.raises(FileNotFoundError, match='absent'):
        orbslam.native_probe()


def test_orbslam_run_uses_selected_command(tmp_path, monkeypatch, no_image_env):
    calls = []
    monkeypatch.setattr(orbslam.subprocess, 'run',
                        lambda cmd, **kw: calls.append((cmd, kw)) or type('R', (), {'returncode': 0})())
    data, export, settings, outdir = paths(tmp_path)
    (outdir).mkdir(parents=True)
    (outdir / 'map_0.txt').write_text('stale')
    assert orbslam.run(data, export, settings, outdir, 'imu') == 0
    assert not (outdir / 'map_0.txt').exists() and (outdir / 'slam.log').exists()
    (cmd, kw), = calls
    assert cmd[2] == 'processing' and kw['env']['UMI_DATA_DIR'] == str(tmp_path.resolve())


def test_slam_cli_requires_an_output_root(monkeypatch, capsys):
    """No default output directory: python -m policy.processing.slam needs --out."""
    from policy.processing.slam import __main__ as cli
    monkeypatch.setattr('sys.argv', ['slam', 'summary'])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2 and '--out' in capsys.readouterr().err


def test_reconstruction_summaries_carry_what_the_cohort_stage_checks(tmp_path):
    """policy.preprocessing.export reads motion_source, result, warnings, quality and provenance.recording_sha256 from
    each summary.json; reconstruct wrote none of them, so every episode was excluded (first as 'Motion summary
    belongs to another estimator', then KeyError 'provenance')."""
    from policy.configs.schema import METHODS
    from policy.data.cloak_dataset import digest
    from policy.preprocessing import MOTION_SOURCES, quality_reasons
    from policy.processing.reconstruct_orbslam import METHOD, cohort_fields
    assert METHOD in METHODS and METHOD in MOTION_SOURCES
    tar = tmp_path / 'recording.tar'
    tar.write_bytes(b'raw recording')
    good = {'plausibility_gate': {'passed': True}, 'data_quality': {'flags': []}}
    assert cohort_fields(good, tar) == {'motion_source': METHOD, 'result': 'initialized', 'warnings': [],
                                        'quality': 'unvalidated', 'provenance': {'recording_sha256': digest(tar)}}
    flagged = cohort_fields({'plausibility_gate': {'passed': False}, 'data_quality': {'flags': ['video_recovered']}},
                            tar)
    assert flagged['warnings'] == ['video_recovered', 'implausible_motion']
    for policy, excluded in (('lenient', False), ('strict', True)):
        reasons = quality_reasons({}, {**good, **flagged}, None, policy)
        assert ('vio_warning_or_uninitialized' in reasons) is excluded


def test_fuse_writes_the_cohort_fields_into_summary_json(tmp_path, monkeypatch):
    """process() (python -m policy.processing.slam fuse|all, and reconstruct_orbslam) writes summary.json through
    write_summary_json; a failed episode gets result 'failed', which the cohort stage excludes as no_initialized_pose
    (not as 'Motion summary belongs to another estimator')."""
    import json
    from policy.data.cloak_dataset import digest
    from policy.preprocessing import MOTION_SOURCES, quality_reasons
    from policy.processing.slam import __main__ as cli, pipeline, recording
    tar = tmp_path / 'data' / 'ep' / 'recording.tar'
    tar.parent.mkdir(parents=True)
    tar.write_bytes(b'raw recording')
    out = tmp_path / 'recon' / 'ep'
    summary = {'episode': 'ep', 'plausibility_gate': {'passed': True}, 'data_quality': {'flags': []},
               'slam_front_end': {'sources_in_fusion': ['orb_slam3_vi', 'klt_two_view']}, 'path_length_m': 1.5}
    returned = cli.write_summary_json(out, summary, tar)
    written = json.loads((out / 'summary.json').read_text())
    assert written == returned and written['path_length_m'] == 1.5
    assert written['motion_source'] in MOTION_SOURCES and written['result'] == 'initialized'
    assert written['quality'] == 'unvalidated' and written['provenance']['recording_sha256'] == digest(tar)
    assert quality_reasons({}, written, None, 'strict') == ['missing_ik']
    # KLT-only (no --require-orb-slam3, ORB-SLAM3 unavailable): flagged, so the strict policy excludes it.
    klt = {**summary, 'slam_front_end': {'sources_in_fusion': ['python_klt_vo_fallback', 'klt_two_view']}}
    assert cli.write_summary_json(out, klt, tar)['warnings'] == ['orb_slam3_unavailable']

    monkeypatch.setattr(recording, 'DATASET', tar.parent.parent)
    monkeypatch.setattr(pipeline, 'ARTIFACTS', out.parent)
    cli.write_failed_summary('ep', RuntimeError('no map'))
    failed = json.loads((out / 'summary.json').read_text())
    assert failed['failed'] and failed['motion_source'] in MOTION_SOURCES and failed['result'] == 'failed'
    assert failed['error'] == "RuntimeError('no map')" and failed['warnings'] == []


def _two_calibrations(tmp_path):
    """The default camera.json ($UMI_EXCAVATOR_CAMERA) and another one with a different lens (fx 802 vs 535)."""
    import json
    from policy.data import camera
    default = json.loads(camera.camera_path().read_text())
    other = json.loads(json.dumps(default))
    other['intrinsics']['matrix'][0][0] = other['intrinsics']['matrix'][1][1] = 802.0
    other['camera_imu']['offset_s'] = default['camera_imu']['offset_s'] + 0.01
    paths = tmp_path / 'default.json', tmp_path / 'other.json'
    for path, data in zip(paths, (default, other)):
        path.write_text(json.dumps(data))
    return paths


class _Stop(Exception):
    pass


def _capture_lens(monkeypatch, seen):
    """Run process() up to tracking and record the lens and camera-IMU offset it was given."""
    from policy.processing.slam import __main__ as cli

    def tracks_for(rec, cam, work):
        seen.append((cam.fx, cli.load_calibration()[2]))
        raise _Stop
    monkeypatch.setattr(cli, 'load', lambda e: object())
    monkeypatch.setattr(cli, 'tracks_for', tracks_for)


def test_reconstruct_uses_the_lens_of_its_calibration(tmp_path, monkeypatch):
    """reconstruct_orbslam copies --calibration into <output>/calibration.json; the lens (Fisheye) must come from that
    copy too, not from camera_path(), or one calibration's lens is mixed with another's camera-IMU extrinsics."""
    import json
    import shutil
    from policy.processing import reconstruct_orbslam
    from policy.processing.slam import pipeline, recording
    default, other = _two_calibrations(tmp_path)
    monkeypatch.setenv('UMI_EXCAVATOR_CAMERA', str(default))
    output, data = tmp_path / 'recon', tmp_path / 'data'
    output.mkdir(); (data / 'ep').mkdir(parents=True)
    shutil.copy2(other, output / 'calibration.json')            # what main() does with --calibration other.json
    monkeypatch.setattr(pipeline, 'ARTIFACTS', None)
    monkeypatch.setattr(pipeline, 'CALIBRATION', None)
    monkeypatch.setattr(recording, 'DATASET', recording.DATASET)
    seen = []
    _capture_lens(monkeypatch, seen)
    row = reconstruct_orbslam.run_one(('ep', tmp_path / 'work', output, data, False))
    cam = json.loads(other.read_text())
    assert seen == [(802.0, cam['camera_imu']['offset_s'])]
    assert row['status'] == 'failed'
    assert json.loads((output / 'ep' / 'summary.json').read_text())['result'] == 'failed'


def test_slam_cli_uses_the_lens_of_calibration(tmp_path, monkeypatch):
    """python -m policy.processing.slam fuse --calibration other.json: lens and extrinsics both from other.json."""
    import json
    from policy.processing.slam import __main__ as cli, pipeline, recording
    default, other = _two_calibrations(tmp_path)
    monkeypatch.setenv('UMI_EXCAVATOR_CAMERA', str(default))
    monkeypatch.setattr(pipeline, 'ARTIFACTS', None)
    monkeypatch.setattr(pipeline, 'CALIBRATION', None)
    monkeypatch.setattr(recording, 'DATASET', recording.DATASET)
    seen = []
    _capture_lens(monkeypatch, seen)
    out = tmp_path / 'recon'
    monkeypatch.setattr('sys.argv', ['slam', 'fuse', 'ep', '--out', str(out), '--work', str(tmp_path / 'work'),
                                     '--data-root', str(tmp_path / 'data'), '--calibration', str(other)])
    cli.main()
    assert seen == [(802.0, json.loads(other.read_text())['camera_imu']['offset_s'])]
    assert json.loads((out / 'ep' / 'summary.json').read_text())['result'] == 'failed'
