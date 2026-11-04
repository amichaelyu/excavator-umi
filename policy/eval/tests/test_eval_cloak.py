"""policy.training.cloak.evaluator: RunEvaluator / evaluate_run (one python -m policy.eval process per report,
cached reports reused) and experiment.py's use of them."""
import json
import sys

import pytest

from policy.training.cloak import evaluator
from policy.training.cloak.evaluator import RunEvaluator, evaluate_run, report_name


class FakeLaunch:
    """Writes a report instead of starting ``python -m policy.eval checkpoint``; scores by step."""
    def __init__(self, manifest='abc', scores=None):
        self.manifest, self.scores, self.calls = manifest, scores or {}, []

    def __call__(self, cmd, log):
        self.calls.append(cmd)
        output = cmd[cmd.index('--output') + 1]
        step = 'initial' if '--pretrained' in cmd else cmd[cmd.index('--checkpoint') + 1].rsplit('/', 1)[-1]
        seed = int(cmd[cmd.index('--mask-seed') + 1]) if '--mask-seed' in cmd else None
        with open(output, 'w') as f:
            json.dump({'manifest_sha256': self.manifest, 'mask_seed': seed, 'step': step,
                       'episode_mean_normalized_mae': {'policy': self.scores.get(step, 1.)}}, f)


def run_dir_with(tmp_path, steps, mode='baseline'):
    run = tmp_path / 'run'
    for step in steps:
        (run / 'checkpoints' / 'excavator_cloak' / mode / str(step)).mkdir(parents=True)
    return run


def test_report_names():
    assert report_name('initial') == 'eval-initial'
    assert report_name(250, masked=True) == 'eval-250-masked'
    assert report_name(250, split='train') == 'eval-250-train'


def test_command_runs_policy_eval_on_the_first_gpu(tmp_path):
    ev = RunEvaluator('/opt/cloak', 'data', tmp_path, manifest_sha256='abc', env={'CUDA_VISIBLE_DEVICES': '2,3'},
                      split='train', predictions=True)
    assert ev.env['CUDA_VISIBLE_DEVICES'] == '2'
    cmd = ev.command(tmp_path / 'eval-250-masked-train.json', checkpoint='ck/250', mask_seed=5)
    assert cmd[:4] == [sys.executable, '-m', 'policy.eval', 'checkpoint']
    for pair in (['--checkpoint', 'ck/250'], ['--mask-seed', '5'], ['--split', 'train'],
                 ['--predictions', str(tmp_path / 'eval-250-masked-train-predictions.npz')]):
        i = cmd.index(pair[0])
        assert cmd[i:i + 2] == pair
    assert '--pretrained' in RunEvaluator('b', 'd', tmp_path, manifest_sha256='x').command('o.json', pretrained=True)


def test_evaluate_run_selects_best_and_stress_tests_it(tmp_path):
    run = run_dir_with(tmp_path, [250, 500, 750])
    launch = FakeLaunch(scores={'250': .5, '500': .2, '750': .3})
    ev = RunEvaluator('b', 'd', run, manifest_sha256='abc', launch=launch, env={})
    out = evaluate_run(ev, steps=[250, 500, 750], mask_seed=9)
    assert out['best_step'] == 500
    assert sorted(out['reports']) == ['eval-250', 'eval-500', 'eval-500-masked', 'eval-750', 'eval-initial']
    assert sorted(p.stem for p in run.glob('eval-*.json')) == sorted(out['reports'])
    assert len(launch.calls) == 5
    # Second call: every report is cached for this dataset.
    evaluate_run(ev, steps=[250, 500, 750], mask_seed=9)
    assert len(launch.calls) == 5
    # A new stress seed re-runs only the stress test.
    evaluate_run(ev, steps=[250, 500, 750], mask_seed=10)
    assert len(launch.calls) == 6 and '--mask-seed' in launch.calls[-1]


def test_evaluate_run_all_saved_steps_and_stress_options(tmp_path):
    run = run_dir_with(tmp_path, [100, 200])
    launch = FakeLaunch()
    ev = RunEvaluator('b', 'd', run, manifest_sha256='abc', launch=launch, env={})
    out = evaluate_run(ev, steps='all', pretrained=False, stress='all', mask_seed=1)
    assert sorted(out['reports']) == ['eval-100', 'eval-100-masked', 'eval-200', 'eval-200-masked']
    out = evaluate_run(ev, steps=[100], pretrained=False, stress='none')
    assert list(out['reports']) == ['eval-100']
    with pytest.raises(FileNotFoundError, match=r'\[300\]'):
        evaluate_run(ev, steps=[300], stress='none')
    with pytest.raises(ValueError, match='mask seed'):
        evaluate_run(ev, steps=[100])


def test_cached_report_of_another_dataset_is_an_error(tmp_path):
    run = run_dir_with(tmp_path, [250])
    FakeLaunch(manifest='other')(['--pretrained', '--output', str(run / 'eval-initial.json')], None)
    ev = RunEvaluator('b', 'd', run, manifest_sha256='abc', launch=FakeLaunch(), env={})
    with pytest.raises(ValueError, match='another dataset'):
        evaluate_run(ev, steps=[250], stress='none')


def test_mode_must_be_given_when_ambiguous(tmp_path):
    run = run_dir_with(tmp_path, [1], mode='cloak')
    (run / 'checkpoints' / 'excavator_cloak' / 'baseline').mkdir()
    with pytest.raises(ValueError, match='--mode'):
        evaluator.saved_steps(run)
    assert evaluator.saved_steps(run, 'cloak') == ('cloak', [1])


def test_experiment_writes_the_same_reports_and_result(frozen, tmp_path, monkeypatch):
    """experiment.execute with the trainer and evaluator processes faked: eval-initial, eval-<step>,
    eval-<best>-masked and result.json, evaluations via python -m policy.eval checkpoint."""
    from policy.training.cloak import experiment
    from policy.data.cloak_dataset import digest, prepare_out_of_view
    source, _, data = frozen
    prepare_out_of_view(source, data, embodiment='handheld-test', content_box=[0, 0, 224, 224],
                        evidence={'confirmation': 'Synthetic test source is out of view',
                                  'source_manifest_sha256': digest(source / 'manifest.json')})
    run = tmp_path / 'run'
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'schema': 'actathon.cloak.experiment.v1', 'task_group': None,
                                  'data': str(data), 'run_dir': str(run),
                                  'training': {'mode': 'cloak_oov', 'steps': 4, 'save_interval': 2},
                                  'evaluate_steps': [2, 4], 'mask_seed': 3}))
    manifest = digest(data / 'manifest.json')
    calls = []

    def fake_run(cmd, cwd, env, stdout, stderr, check):
        calls.append((cmd[2], cmd[3], env.get('CUDA_VISIBLE_DEVICES')))
        if cmd[2] == 'policy.training.cloak' and '--check-data' not in cmd:
            for step in (2, 4):
                (run / 'checkpoints/excavator_cloak/cloak_oov' / str(step)).mkdir(parents=True)
        elif cmd[2] == 'policy.eval':
            FakeLaunch(manifest, {'initial': .9, '2': .3, '4': .4})(cmd, None)

    monkeypatch.setattr('subprocess.run', fake_run)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    result = experiment.execute(recipe, tmp_path, tmp_path)
    assert sorted(p.name for p in run.glob('eval-*.json')) == ['eval-2-masked.json', 'eval-2.json', 'eval-4.json',
                                                               'eval-initial.json']
    assert sorted(p.name for p in run.glob('*.log')) == ['data-check.log', 'eval-2-masked.log', 'eval-2.log',
                                                         'eval-4.log', 'eval-initial.log', 'training.log']
    assert result['selected_step'] == 2 and result['initial'] == {'policy': .9}
    assert result['selected'] == {'policy': .3} and result['masked'] == {'policy': .3}
    assert json.loads((run / 'result.json').read_text()) == result
    assert [c[:2] for c in calls] == [('policy.training.cloak', 'train'), ('policy.eval', 'checkpoint'),
                                      ('policy.training.cloak', 'train')] + [('policy.eval', 'checkpoint')] * 3
    assert {c[2] for c in calls if c[0] == 'policy.eval'} == {'0'} and calls[0][2] == '0,1'
