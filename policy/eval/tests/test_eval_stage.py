"""policy.eval.stage: the evaluate config section (policy.configs.schema.check_evaluate), run() against a
recording fake Host, skipping reports."""
import hashlib
import json
import sys

import pytest
import yaml

from policy import configs
from policy.configs.schema import check_evaluate
from policy.eval import stage
from policy.training.lambda_host import CODE

ROOT = stage.ROOT


class FakeHost:
    """Records Host calls; ``remote`` maps report names the 'box' writes into the run dir on docker_run."""
    def __init__(self, dry_run=False, running='', write=None):
        self.address, self.dry_run, self.calls, self.running, self.write = 'ubuntu@203.0.113.7', dry_run, [], running, write

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == 'ssh':
                return self.running
            if name == 'docker_run' and args[0].startswith('eval-') and self.write:
                self.write()
            return 'umi-excavator-train:test' if name == 'ensure_image' else None
        return record

    def names(self):
        return [c[0] for c in self.calls]

    def docker(self, prefix):
        return [c for c in self.calls if c[0] == 'docker_run' and c[1][0].startswith(prefix)]


def cohort(cfg, holdout=None):
    folder = ROOT / cfg.cohort_data
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps({'schema': 'actathon.cloak.v1', 'episodes': [],
                                                      **({'holdout': holdout} if holdout else {})}))
    return hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest()


# check_evaluate ----------------------------------------------------------------------------------------

def test_every_shipped_config_section_is_valid():
    for path in (ROOT / 'configs/policy').glob('*.yaml'):
        check_evaluate(yaml.safe_load(path.read_text()).get('evaluate') or {})


def test_full_section_is_valid():
    check_evaluate({'every': 250, 'steps': [250, 1000], 'pretrained': False, 'mask_seed': 0, 'stress': 'all',
                    'split': 'train', 'predictions': True, 'runner': 'local', 'gpus': [1],
                    'backend': '/opt/cloak'})
    check_evaluate({'steps': 'all'})
    check_evaluate(None)


@pytest.mark.parametrize('section, message', [
    ({'horizon': 10}, 'unknown evaluate options'),
    ({'every': 0}, 'every'),
    ({'every': True}, 'every'),
    ({'steps': []}, 'steps'),
    ({'steps': [250, 250]}, 'repeats'),
    ({'steps': 'best'}, 'steps'),
    ({'pretrained': 'yes'}, 'pretrained'),
    ({'mask_seed': -1}, 'mask_seed'),
    ({'stress': 'worst'}, 'stress'),
    ({'split': 'test'}, 'split'),
    ({'runner': 'modal'}, 'runner'),
    ({'gpus': []}, 'gpus'),
    ({'gpus': 0}, 'gpus'),
])
def test_invalid_sections(section, message):
    with pytest.raises(ValueError, match=message):
        check_evaluate(section)


def test_defaults_follow_the_train_recipe(make_config):
    cfg = make_config(train={**configs.load('dirt-orb').section('train'), 'seed': 5},
                      **{'lambda': {'host': 'ubuntu@x', 'gpus': [3, 4]}})
    opts = stage.options(cfg)
    assert opts['steps'] == cfg.recipe()['evaluate_steps'] == [250, 500, 750, 1000]
    assert opts['mask_seed'] == cfg.recipe()['mask_seed'] == 5
    assert (opts['pretrained'], opts['stress'], opts['split'], opts['runner'], opts['gpus']) == \
        (True, 'best', 'holdout', 'lambda', [3])
    assert stage.options(cfg, steps=[500, 250])['steps'] == [250, 500]
    assert stage.options(cfg, steps=['all'])['steps'] == 'all'
    with pytest.raises(ValueError, match='beyond train.steps'):
        stage.options(cfg, steps=[2000])


def test_twin_mask_cohort_skips_the_stress_test(make_config, report_file):
    """cloak / noaug train on twin-rendered masks (cfg.masks twin): no synthetic edge-mask stress test, which needs
    an out-of-view source. The plan then needs no masked report, and the result says why."""
    cfg = make_config(train={**configs.load('dirt-orb').section('train'), 'mode': 'cloak'},
                      evaluate={'steps': [250, 500], 'stress': 'all'})
    assert cfg.masks == 'twin'
    opts = stage.options(cfg)
    assert opts['stress'] == 'none' and 'out-of-view' in opts['stress_skipped']
    assert '--stress none' in ' '.join(stage.command(cfg, opts))
    manifest = cohort(cfg)
    run = ROOT / cfg.run_dir
    for name, policy in [('initial', .9), ('250', .3), ('500', .4)]:
        report_file(run / f'eval-{name}.json', manifest, policy)
    host = FakeHost()
    out = stage.run(cfg, host)
    assert host.names() == ['pull']                 # nothing missing: no masked report is expected
    assert set(out['reports']) == {'eval-initial', 'eval-250', 'eval-500'}
    assert out['stress_skipped'] == opts['stress_skipped'] and out['summary']['best']['step'] == 250
    assert 'stress_skipped' not in stage.run(make_config(evaluate={'steps': [250]}), FakeHost(dry_run=True))


# run ---------------------------------------------------------------------------------------------------

def test_dry_run_on_a_host_records_the_commands(make_config):
    cfg = make_config()
    host = FakeHost(dry_run=True)
    out = stage.run(cfg, host, dry_run=True)
    assert host.names() == ['pull', 'push_code', 'push_data', 'ensure_image', 'docker_run', 'docker_run', 'pull']
    assert host.calls[2][1] == (cfg.cohort_data,)
    assert host.calls[4][1][0] == 'fetch-pi05-base'
    (name, cmd), kwargs = host.calls[5][1], host.calls[5][2]
    assert name == 'eval-dirt-orb' and kwargs == {'gpus': [0], 'detach': False, 'replace': True}
    assert cmd[:4] == ['python', '-m', 'policy.eval', 'run']
    joined = ' '.join(cmd)
    assert f'--data {cfg.cohort_data} --run-dir {cfg.run_dir}' in joined and '--backend /opt/cloak' in joined
    assert '--mode baseline --steps 250 500 750 1000 --stress best --mask-seed 20261006' in joined
    assert '--no-pretrained' not in joined and '--split' not in joined
    assert set(out['reports']) == {'eval-initial', 'eval-250', 'eval-500', 'eval-750', 'eval-1000'}
    assert {r['status'] for r in out['reports'].values()} == {'planned'}
    assert 'policy/eval' in CODE and 'policy/data' in CODE


def test_skips_when_every_report_exists_for_the_same_dataset(make_config, report_file):
    cfg = make_config(evaluate={'steps': [250, 500]})
    manifest = cohort(cfg)
    run = ROOT / cfg.run_dir
    for name, policy in [('initial', .9), ('250', .3), ('500', .4)]:
        report_file(run / f'eval-{name}.json', manifest, policy)
    report_file(run / 'eval-250-masked.json', manifest, .35, mask_seed=20261006)
    host = FakeHost()
    out = stage.run(cfg, host)
    assert host.names() == ['pull']
    assert {n: r['status'] for n, r in out['reports'].items()} == {
        'eval-initial': 'existing', 'eval-250': 'existing', 'eval-500': 'existing', 'eval-250-masked': 'existing'}
    assert out['summary']['best']['step'] == 250 and 'eval-250 *' in out['summary']['table']


def test_runs_what_is_missing_and_reports_new_files(make_config, report_file):
    cfg = make_config(evaluate={'steps': [250, 500], 'pretrained': False, 'gpus': [1]})
    manifest = cohort(cfg)
    run = ROOT / cfg.run_dir
    report_file(run / 'eval-250.json', manifest, .3)
    # A stress report with another seed is stale, not reused.
    report_file(run / 'eval-250-masked.json', manifest, .3, mask_seed=1)

    def box():
        report_file(run / 'eval-500.json', manifest, .2)
        report_file(run / 'eval-500-masked.json', manifest, .25, mask_seed=20261006)
    host = FakeHost(write=box)
    out = stage.run(cfg, host)
    assert not host.docker('fetch-pi05-base') and host.docker('eval-')[0][2]['gpus'] == [1]
    assert '--no-pretrained' in host.docker('eval-')[0][1][1]
    status = {n: r['status'] for n, r in out['reports'].items()}
    assert status == {'eval-250': 'existing', 'eval-500': 'new', 'eval-500-masked': 'new'}
    assert out['reports']['eval-500']['path'] == str(cfg.run_dir / 'eval-500.json')


def test_report_of_another_dataset_is_an_error(make_config, report_file):
    cfg = make_config()
    cohort(cfg)
    report_file(ROOT / cfg.run_dir / 'eval-initial.json', 'not-this-cohort', .9)
    with pytest.raises(ValueError, match='another dataset'):
        stage.run(cfg, FakeHost())


def test_refuses_the_training_gpus_while_training_runs(make_config):
    cfg = make_config()
    with pytest.raises(RuntimeError, match='still running'):
        stage.run(cfg, FakeHost(running='abc123\n'))
    host = FakeHost(running='abc123\n')
    stage.run(make_config(evaluate={'gpus': [1]}), host)
    assert host.docker('eval-')


def test_no_holdout_cohort_needs_the_train_split(make_config):
    cfg = make_config()
    cohort(cfg, holdout='none')
    with pytest.raises(ValueError, match='holdout: none'):
        stage.run(cfg, FakeHost(dry_run=True))
    host = FakeHost(dry_run=True)
    out = stage.run(make_config(evaluate={'split': 'train', 'stress': 'none'}), host, steps=[1000])
    assert '--split train' in ' '.join(host.docker('eval-')[0][1][1])
    assert set(out['reports']) == {'eval-initial-train', 'eval-1000-train'}


def test_local_runner_uses_this_python(make_config, capsys):
    cfg = make_config(evaluate={'runner': 'local', 'backend': 'dataset/cloak-research/upstream'})
    out = stage.run(cfg, None, steps='all', dry_run=True)
    cmd = [l for l in capsys.readouterr().err.splitlines() if l.startswith('+ ')][0]
    assert cmd.startswith(f'+ {sys.executable} -m policy.eval run --backend dataset/cloak-research/upstream')
    assert '--steps all' in cmd
    assert set(out['reports']) == {'eval-initial'}


def test_local_runner_needs_a_trained_run(make_config):
    cfg = make_config(evaluate={'runner': 'local'})
    with pytest.raises(FileNotFoundError, match='train first'):
        stage.run(cfg, None)

