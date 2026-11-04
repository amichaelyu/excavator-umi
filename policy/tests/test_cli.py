"""python -m policy: stage order, `all`, config names, runners per stage, eval delegation, `check`, dry runs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from policy import __main__ as cli
from policy import configs, stages
from policy.training import lambda_host
from policy.training import stages as training

ROOT = Path(__file__).resolve().parents[2]
DIRT = ROOT / 'configs/policy/dirt-orb.yaml'


def write(tmp_path, file='cfg', **sections):
    """dirt-orb with ``sections`` merged in (a None value removes the key)."""
    raw = yaml.safe_load(DIRT.read_text())
    for section, values in sections.items():
        if isinstance(values, dict):
            merged = {**(raw.get(section) or {}), **values}
            raw[section] = {k: v for k, v in merged.items() if v is not None}
        else:
            raw[section] = values
    path = tmp_path / f'{file}.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path


def commands(capsys):
    return [line[2:] for line in capsys.readouterr().err.splitlines() if line.startswith('+ ')]


@pytest.fixture
def fake_eval(monkeypatch):
    """policy.eval.stage.run, recorded (the orchestrator imports the eval stage as ``stages.evaluation``)."""
    calls = SimpleNamespace(run=[])

    def run(cfg, host, steps=None, dry_run=False):
        calls.run.append((cfg.name, host, steps, dry_run))
        return {'evaluated': steps or 'all'}

    monkeypatch.setattr(stages, 'evaluation', SimpleNamespace(run=run))
    return calls


BOX = 'ubuntu@203.0.113.10'


@pytest.fixture(autouse=True)
def lambda_box(tmp_path, monkeypatch):
    """The Lambda box comes from this machine's configs/hub/local.json (hosts.lambda); here a temporary one."""
    local = tmp_path / 'local.json'
    local.write_text(json.dumps({'hosts': {'lambda': {'host': BOX}}}))
    monkeypatch.setenv('UMI_EXCAVATOR_LOCAL_CONFIG', str(local))
    return BOX


@pytest.fixture
def no_push(monkeypatch):
    monkeypatch.setattr(lambda_host.Host, 'push_code', lambda self, code=lambda_host.CODE: None)


# --- stages and order ---------------------------------------------------------------------------------------------

def test_pipeline_order():
    assert stages.PIPELINE == ('reconstruct', 'labels', 'cohort', 'train', 'eval')
    assert stages.STAGES == (*stages.PIPELINE, 'serve')
    assert set(stages.COMMANDS) == {*stages.STAGES, 'all', 'status', 'stop', 'recipe', 'image', 'check'}
    assert set(stages.STAGE_FUNCTIONS) == set(stages.COMMANDS) - {'all', 'check'}


def test_all_runs_the_pipeline_in_order(monkeypatch):
    calls = []
    for stage in stages.STAGE_FUNCTIONS:
        monkeypatch.setitem(stages.STAGE_FUNCTIONS, stage,
                            lambda r, in_all=False, stage=stage: calls.append((stage, in_all)))
    assert cli.main(['all', 'dirt-orb']) == 0
    assert calls == [(stage, True) for stage in stages.PIPELINE]


def test_single_stage_runs_only_itself(monkeypatch):
    calls = []
    monkeypatch.setitem(stages.STAGE_FUNCTIONS, 'cohort', lambda r, in_all=False: calls.append((r.cfg.name, in_all)))
    assert cli.main(['cohort', 'dirt-orb']) == 0
    assert calls == [('dirt-orb', False)]


def test_all_dry_run_prints_every_stage(capsys, no_push, fake_eval):
    assert cli.main(['all', 'dirt-orb', '--dry-run']) == 0
    err = capsys.readouterr().err
    banners = [line for line in err.splitlines() if line.startswith('== ')]
    assert [b.split()[1] for b in banners] == list(stages.PIPELINE)
    order = [err.index(s) for s in ('-m policy.processing.reconstruct_orbslam', '-m digital_twin.sim.replay',
                                    '# export dataset/cohorts/dirt-orb/selection.json', 'policy.training.cloak.experiment',
                                    '# wait for train-dirt-orb')]
    assert order == sorted(order)
    (name, host, steps, dry_run), = fake_eval.run
    assert (name, host.address, host.root, steps, dry_run) == ('dirt-orb', BOX, '~/umi-excavator', None, True)


# --- config names -------------------------------------------------------------------------------------------------

def test_config_names_and_paths_resolve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for given in ('dirt-orb', 'dirt-orb.yaml', 'configs/policy/dirt-orb.yaml', str(DIRT)):
        assert configs.load(given).path == DIRT
    other = write(tmp_path, 'mine', name='mine')
    assert configs.load(other).name == 'mine'
    assert configs.load('mine.yaml').name == 'mine'              # relative to the working directory


def test_unknown_config_lists_the_names(capsys):
    assert cli.main(['check', 'no-such-config']) == 2
    err = capsys.readouterr().err
    assert 'No config no-such-config' in err and 'dirt-orb' in err


# --- runners ------------------------------------------------------------------------------------------------------

def test_labels_and_cohort_run_locally_by_default(tmp_path, capsys):
    cfg = configs.load(write(tmp_path))
    stages.run(cfg, 'labels', dry_run=True)
    cmds = commands(capsys)
    assert len(cmds) == 1 and '-m digital_twin.sim.replay' in cmds[0] and 'docker' not in cmds[0]


@pytest.mark.parametrize('stage', ['labels', 'cohort'])
def test_docker_runner_from_the_config(tmp_path, capsys, stage):
    path = write(tmp_path, **{stage: {'runner': 'docker'}})
    stages.run(configs.load(path), stage, dry_run=True)
    build, run = commands(capsys)
    assert build.startswith('docker build -f') and 'preprocess.Dockerfile' in build
    assert f'-v {ROOT}:/workspace' in run and f'-v {path}:{path}:ro' in run
    assert run.endswith(f'python -m policy {stage} {path} --runner local')


def test_cli_runner_overrides_the_config(tmp_path, capsys):
    path = write(tmp_path, labels={'runner': 'docker'})
    assert cli.main(['labels', str(path), '--runner', 'local', '--dry-run']) == 0
    assert not any('docker' in c for c in commands(capsys))
    assert cli.main(['labels', str(path), '--docker', '--dry-run']) == 0
    assert any(c.startswith('docker run') for c in commands(capsys))


def test_reconstruct_runner_override(capsys):
    assert cli.main(['reconstruct', 'dirt-orb', '--runner', 'docker', '--dry-run']) == 0
    build, run = commands(capsys)
    tag = lambda_host.processing_image_tag()
    assert tag in build and run.startswith('docker run') and f'{tag} python -m policy.processing.reconstruct_orbslam' in run


@pytest.mark.parametrize('argv, message', [
    (['labels', 'dirt-orb', '--runner', 'lambda'], "labels runs on one of ('local', 'docker')"),
    (['train', 'dirt-orb', '--runner', 'docker'], 'train has no runner choice'),
    (['all', 'dirt-orb', '--docker'], '--runner overrides one stage'),
])
def test_bad_runner_overrides(capsys, argv, message):
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


def test_runner_per_stage(tmp_path):
    cfg = configs.load(write(tmp_path, reconstruct={'runner': 'lambda'}, cohort={'runner': 'docker'},
                             evaluate={'runner': 'local'}))
    assert [cfg.runner(s) for s in stages.STAGES] == ['lambda', 'local', 'docker', 'lambda', 'local', 'lambda']


@pytest.mark.parametrize('sections, message', [
    ({'labels': {'runner': 'lambda'}}, 'labels.runner'),
    ({'cohort': {'runner': 'k8s'}}, 'cohort.runner'),
    ({'evaluate': {'runner': 'docker'}}, 'evaluate.runner'),
    ({'labels': {'colour': 'red'}}, 'unknown labels options'),
    ({'train': {'mode': 'cloak', 'steps': 10}, 'cohort': {'masks': 'out_of_view'}}, 'cohort.masks'),
    ({'train': {'mode': 'cloak_oov'}, 'cohort': {'masks': 'twin'}}, 'cohort.masks'),
    ({'train': {'stop_at': 5}}, 'unknown train options'),
    ({'source': {'visibility': None}}, "source needs ['visibility']"),
    ({'source': {'recordings': {'data_root': 'dataset'}}}, 'source.recordings must be a list'),
    ({'source': {'recordings': [{'label': 'dirt digging'}]}}, 'needs a data_root'),
    ({'source': {'recordings': [{'data_root': 'dataset'}]}}, 'needs a label or ids'),
    ({'source': {'recordings': [{'data_root': 'dataset', 'glob': '*'}]}}, 'unknown source.recordings options'),
    ({'source': {'holdout': 1.0}}, 'source.holdout'),
    ({'source': {'selection': 'dataset/handheld/dirt.json'}}, 'unknown source options'),
    ({'serve': {'step': 'latest'}}, 'serve.step'),
])
def test_rejects_bad_sections(tmp_path, sections, message):
    with pytest.raises(ValueError, match=message.replace('[', r'\[').replace(']', r'\]').replace('(', r'\(')):
        configs.load(write(tmp_path, **sections))


def test_reconstruction_uses_the_camera_config(tmp_path, capsys):
    """The reconstruction takes configs/hardware/camera-calibration/camera.json (policy.data.camera); source.calibration is not a key."""
    cfg = configs.load(write(tmp_path))
    assert cfg.calibration == Path('configs/hardware/camera-calibration/camera.json')
    training.reconstruct(cfg, dry_run=True)
    assert '--calibration configs/hardware/camera-calibration/camera.json' in commands(capsys)[-1]
    with pytest.raises(ValueError, match='unknown source options'):
        configs.load(write(tmp_path, source={'calibration': 'dataset/calibration.json'}))


# --- eval delegation ----------------------------------------------------------------------------------------------

def test_load_validates_the_evaluate_section(tmp_path):
    cfg = configs.load(write(tmp_path, evaluate={'every': 250, 'stress': 'all', 'split': 'train'}))
    assert cfg.evaluate_steps() == [250, 500, 750, 1000]
    with pytest.raises(ValueError, match='unknown evaluate options'):
        configs.load(write(tmp_path, evaluate={'bad': True}))
    with pytest.raises(ValueError, match='evaluate.stress'):
        configs.load(write(tmp_path, evaluate={'stress': 'worst'}))


def test_eval_on_lambda_gets_the_host(capsys, fake_eval):
    assert cli.main(['eval', 'dirt-orb', '--host', 'ubuntu@203.0.113.7', '--steps', '250', '500']) == 0
    (name, host, steps, dry_run), = fake_eval.run
    assert isinstance(host, lambda_host.Host) and host.address == 'ubuntu@203.0.113.7'
    assert (steps, dry_run) == ([250, 500], False)
    assert json.loads(capsys.readouterr().out) == {'evaluated': [250, 500]}


def test_eval_steps_all(capsys, fake_eval):
    assert cli.main(['eval', 'dirt-orb', '--steps', 'all', '--dry-run']) == 0
    assert fake_eval.run[-1][2:] == ('all', True)
    with pytest.raises(SystemExit):
        cli.main(['eval', 'dirt-orb', '--steps', 'all', '250'])
    with pytest.raises(SystemExit):
        cli.main(['eval', 'dirt-orb', '--steps', 'best'])


def test_eval_locally_gets_no_host(tmp_path, fake_eval):
    path = write(tmp_path, evaluate={'runner': 'local'}, **{'lambda': {'host': None}})
    assert cli.main(['eval', str(path), '--dry-run']) == 0
    assert fake_eval.run == [('dirt-orb', None, None, True)]


# --- check --------------------------------------------------------------------------------------------------------

def test_check_prints_the_plan(capsys):
    assert cli.main(['check', 'dirt-orb']) == 0
    out = capsys.readouterr().out
    assert 'dirt-orb: kind cloak, task dirt' in out and 'config OK' in out
    lines = [line.split()[1] for line in out.splitlines() if line[:1].isdigit()]
    assert lines == list(stages.STAGES)
    assert 'reconstruct local: this machine' in out
    assert f'Docker on {BOX}:~/umi-excavator, umi-excavator-train image, GPUs [0]' in out
    assert 'out  dataset/cohorts/dirt-orb/frozen/orbslam3_smoother' in out
    assert 'in   dataset/reconstructions/dirt-orb  (from reconstruct)' in out
    assert 'holdout eval at [250, 500, 750, 1000]' in out


def test_check_follows_runners_masks_and_host(tmp_path, capsys):
    path = write(tmp_path, reconstruct={'runner': 'lambda'}, labels={'runner': 'docker'}, train={'mode': 'cloak'})
    assert cli.main(['check', str(path), '--host', 'ubuntu@203.0.113.7']) == 0
    out = capsys.readouterr().out
    assert 'reconstruct lambda: Docker on ubuntu@203.0.113.7:~/umi-excavator, umi-excavator-processing image' in out
    assert 'labels      docker: Docker on this machine, umi-excavator-preprocess image' in out
    assert 'twin embodiment masks' in out and 'frozen-twin/orbslam3_smoother' in out


# --- dry runs -----------------------------------------------------------------------------------------------------

def test_cohort_with_twin_masks(tmp_path, capsys):
    cfg = configs.load(write(tmp_path, train={'mode': 'noaug'}, cohort={'mask_workers': 3}))
    assert cfg.masks == 'twin' and cfg.cohort_data == Path('dataset/cohorts/dirt-orb/frozen-twin/orbslam3_smoother')
    stages.run(cfg, 'cohort', dry_run=True)
    err = capsys.readouterr().err
    cmds = [line[2:] for line in err.splitlines() if line.startswith('+ ')]
    assert '# export dataset/cohorts/dirt-orb/selection.json -> dataset/cohorts/dirt-orb/export' in err
    assert cmds[0].endswith('-m policy.training.cloak.twin_masks --data dataset/cohorts/dirt-orb/export/source '
                            '--output dataset/cohorts/dirt-orb/export/twin-masks --workers 3')
    assert ' -m policy.training.cloak prepare ' in cmds[1] and '--output dataset/cohorts/dirt-orb/export/prepared-twin' in cmds[1]
    assert '# freeze dataset/cohorts/dirt-orb/export/prepared-twin -> ' \
           'dataset/cohorts/dirt-orb/frozen-twin/orbslam3_smoother' in err
    assert cfg.recipe()['data'] == str(cfg.cohort_data)


def test_train_dry_run_writes_nothing(tmp_path, capsys, no_push):
    run_dir = tmp_path / 'run'
    cfg = configs.load(write(tmp_path, train={'run_dir': str(run_dir), 'microbatch': 2,
                                              'initial_checkpoint': 'dataset/runs/x/checkpoints/250'}))
    stages.run(cfg, 'train', dry_run=True, host='ubuntu@203.0.113.7')
    assert not run_dir.exists()
    err = capsys.readouterr().err
    assert '"microbatch": 2' in err and '"initial_checkpoint"' in err
    run = next(c for c in err.splitlines() if 'docker run --name train-dirt-orb' in c)
    assert 'policy.training.cloak.experiment --recipe' in run and '--gpus' in run


def test_train_skips_a_finished_run(tmp_path, capsys):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'result.json').write_text('{"selected_step": 500}')
    stages.run(configs.load(write(tmp_path, train={'run_dir': str(run_dir)})), 'train', dry_run=True)
    err = capsys.readouterr().err
    assert 'already trained' in err and 'docker run' not in err


def test_wait_follows_the_container(tmp_path, monkeypatch):
    run_dir = tmp_path / 'run'
    cfg = configs.load(write(tmp_path, train={'run_dir': str(run_dir)}, **{'lambda': {'poll': 1}}))
    states = iter(['running 0', 'running 0', 'exited 0'])
    pulled = []

    class Box:
        address, dry_run = 'ubuntu@203.0.113.7', False

        def ssh(self, command, check=True, capture=False):
            return next(states)

        def pull(self, path):
            pulled.append(path)
            run_dir.mkdir(exist_ok=True)
            (run_dir / 'result.json').write_text('{}')

    monkeypatch.setattr(training.time, 'sleep', lambda s: None)
    training.wait_for(cfg, Box())
    assert pulled == [run_dir]
    states = iter(['exited 1'])
    (run_dir / 'result.json').unlink()
    with pytest.raises(RuntimeError, match='exited with code 1'):
        training.wait_for(cfg, SimpleNamespace(address='box', dry_run=False, ssh=lambda *a, **k: next(states),
                                               pull=lambda path: None))


def test_eval_prints_the_summary_table(monkeypatch, capsys):
    result = {'reports': {'eval-250': {'status': 'new'}}, 'summary': {'table': 'step  policy\n250   0.123'}}
    monkeypatch.setattr(stages, 'evaluation', SimpleNamespace(run=lambda *a, **k: result))
    assert cli.main(['eval', 'dirt-orb', '--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'eval-250' in out and 'new' in out and '250   0.123' in out
