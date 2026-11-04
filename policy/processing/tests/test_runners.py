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
