"""reconstruct.runner: local Python, the processing image via Docker here, or on a Lambda box (dry runs)."""
from pathlib import Path

import pytest
import yaml

from policy import configs
from policy.training import lambda_host, stages

ROOT = Path(__file__).resolve().parents[3]


def load(tmp_path, **reconstruct):
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw['reconstruct'].update(reconstruct)
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(raw))
    return configs.load(path)


def commands(capsys):
    return [line[2:] for line in capsys.readouterr().err.splitlines() if line.startswith('+ ')]


def test_local_is_the_default(tmp_path, capsys):
    stages.reconstruct(load(tmp_path), dry_run=True)
    cmd, = commands(capsys)
    assert ' -m policy.processing.reconstruct_orbslam ' in cmd and 'docker' not in cmd


def test_docker_runs_the_processing_image_with_the_repository(tmp_path, capsys):
    stages.reconstruct(load(tmp_path, runner='docker', workers=3), dry_run=True)
    build, run = commands(capsys)
    tag = lambda_host.processing_image_tag()
    assert tag in build and 'docker build -f policy/processing/docker/Dockerfile' in build
    assert f'-v {ROOT}:/workspace -w /workspace {tag} python -m policy.processing.reconstruct_orbslam' in run
    assert '--data-root dataset' in run and '--workers 3' in run and '--no-video' not in run


def test_docker_mounts_inputs_outside_the_repository(tmp_path, capsys):
    cfg = load(tmp_path, runner='docker')
    cfg.raw['source']['recordings'] = [{'data_root': str(tmp_path / 'recordings'), 'label': 'dirt digging'}]
    stages.reconstruct(cfg, dry_run=True)
    run = commands(capsys)[-1]
    real = (tmp_path / 'recordings').resolve()
    assert f'-v {real}:{real}:ro' in run and f'--data-root {real}' in run


def test_lambda_builds_and_runs_on_the_box(tmp_path, capsys, monkeypatch):
    pushed = []
    monkeypatch.setattr(lambda_host.Host, 'push_code', lambda self, code=lambda_host.CODE: pushed.append(code))
    stages.reconstruct(load(tmp_path, runner='lambda'), dry_run=True, address='ubuntu@203.0.113.7')
    cmds = commands(capsys)
    assert pushed == [lambda_host.PROCESSING_CODE]
    assert any('rsync' in c and 'dataset/' in c for c in cmds)
    assert any(lambda_host.processing_build_command(lambda_host.processing_image_tag()) in c for c in cmds)
    run = next(c for c in cmds if 'docker run' in c)
    assert 'ubuntu@203.0.113.7' in run and '-w /workspace' in run and '/cache/openpi' not in run
    assert 'python -m policy.processing.reconstruct_orbslam' in run
    assert 'rsync' in cmds[-1] and cmds[-1].endswith('dataset/reconstructions/dirt-orb/')


def test_unknown_runner_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='reconstruct.runner'):
        load(tmp_path, runner='k8s')
