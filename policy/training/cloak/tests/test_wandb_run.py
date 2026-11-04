"""policy.training.cloak.wandb_run against a stub ``wandb`` module (the real package is only in the cloak
dependency group, so these pass with or without it installed)."""
import json
import sys
import types

import pytest

from policy.training.cloak import wandb_run

SETTINGS = {'project': 'umi-excavator', 'entity': 'team', 'mode': 'online', 'tags': ['dirt'], 'name': 'dirt-orb',
            'group': 'dirt-orb'}
KEY = {'WANDB_API_KEY': 'k' * 40}
REPORT = {'episode_mean_normalized_mae': {'policy': .3, 'repeat_state': .5, 'training_mean': .9}}


class StubRun:
    def __init__(self, calls):
        self.calls = calls

    def log(self, data, step=None):
        self.calls.append(('log', data, step))

    def define_metric(self, name, step_metric=None):
        self.calls.append(('define_metric', name, step_metric))

    def finish(self, exit_code=None):
        self.calls.append(('finish', exit_code))


@pytest.fixture
def stub(monkeypatch):
    """A fake ``wandb`` in sys.modules; returns the list of calls made on it."""
    calls = []
    module = types.ModuleType('wandb')

    def init(**kwargs):
        calls.append(('init', kwargs))
        return StubRun(calls)

    module.init = init
    monkeypatch.setitem(sys.modules, 'wandb', module)
    monkeypatch.delenv('WANDB_MODE', raising=False)
    return calls


@pytest.fixture
def no_wandb(monkeypatch):
    """Any ``import wandb`` raises ImportError."""
    monkeypatch.setitem(sys.modules, 'wandb', None)
    monkeypatch.delenv('WANDB_MODE', raising=False)


def test_init_log_finish_with_a_stable_id(tmp_path, stub):
    record = wandb_run.write(tmp_path, SETTINGS)
    assert record == {'id': record['id'], **SETTINGS} and len(record['id']) == 12
    with wandb_run.session(tmp_path, config={'steps': 10}, environ=KEY) as run:
        run.log({'loss': 1.5}, step=1)
    assert stub == [('init', {'project': 'umi-excavator', 'entity': 'team', 'name': 'dirt-orb',
                              'group': 'dirt-orb', 'tags': ['dirt'], 'mode': 'online', 'config': {'steps': 10},
                              'dir': str(tmp_path), 'id': record['id'], 'resume': 'allow'}),
                    ('log', {'loss': 1.5}, 1), ('finish', 0)]
    # Rewriting the settings (a resumed `train` with other tags) keeps the id; the resumed process reuses it.
    assert wandb_run.write(tmp_path, {**SETTINGS, 'tags': []})['id'] == record['id']
    stub.clear()
    with wandb_run.session(tmp_path, environ=KEY):
        pass
    assert stub[0][1]['id'] == record['id'] and stub[0][1]['tags'] is None and stub[0][1]['config'] is None
    assert 'k' * 40 not in (tmp_path / wandb_run.FILE).read_text()


def test_a_record_without_an_id_gets_one_and_keeps_it(tmp_path, stub):
    (tmp_path / wandb_run.FILE).write_text(json.dumps({'mode': 'offline'}))
    with wandb_run.session(tmp_path, environ={}):
        pass
    saved = wandb_run.load(tmp_path)['id']
    assert stub[0][1]['id'] == saved and stub[0][1]['mode'] == 'offline' and stub[0][1]['project'] == 'umi-excavator'


def test_an_error_finishes_the_run_as_failed(tmp_path, stub):
    wandb_run.write(tmp_path, SETTINGS)
    with pytest.raises(FloatingPointError), wandb_run.session(tmp_path, environ=KEY):
        raise FloatingPointError('nonfinite loss')
    assert stub[-1] == ('finish', 1)


def test_disabled_makes_no_calls(tmp_path, no_wandb):
    with wandb_run.session(tmp_path, environ=KEY) as run:           # no wandb-run.json
        assert run is None
    assert wandb_run.check(None, KEY) == 'disabled'
    wandb_run.write(tmp_path, {**SETTINGS, 'mode': 'disabled'})
    with wandb_run.session(tmp_path, environ=KEY) as run:
        assert run is None
    assert wandb_run.log_eval(tmp_path, REPORT, step=250, environ=KEY) is None
    wandb_run.write(tmp_path, SETTINGS)                              # on, but WANDB_MODE=disabled for this launch
    with wandb_run.session(tmp_path, environ={**KEY, 'WANDB_MODE': 'disabled'}) as run:
        assert run is None


def test_mode_overrides(stub):
    assert wandb_run.mode(SETTINGS, {}) == 'online'
    assert wandb_run.mode(SETTINGS, {'WANDB_MODE': 'offline'}) == 'offline'
    assert wandb_run.mode(SETTINGS, {'WANDB_MODE': 'dryrun'}) == 'offline'
    # WANDB_MODE never turns tracking on for a run that does not track.
    assert wandb_run.mode({**SETTINGS, 'mode': 'disabled'}, {'WANDB_MODE': 'online'}) == 'disabled'
    assert wandb_run.check(SETTINGS, {'WANDB_MODE': 'offline'}) == 'offline'     # offline needs no key


def test_check_fails_fast(stub, monkeypatch):
    with pytest.raises(RuntimeError, match='WANDB_API_KEY'):
        wandb_run.check(SETTINGS, {})
    with pytest.raises(RuntimeError, match='WANDB_MODE'):
        wandb_run.check(SETTINGS, {**KEY, 'WANDB_MODE': 'cloud'})
    assert wandb_run.check(SETTINGS, KEY) == 'online'
    monkeypatch.setitem(sys.modules, 'wandb', None)
    with pytest.raises(RuntimeError, match='not installed'):
        wandb_run.check(SETTINGS, KEY)
    assert wandb_run.check(SETTINGS, KEY, package=False) == 'online'    # the launching Mac has no wandb


def test_log_eval_rows_and_axis(tmp_path, stub):
    record = wandb_run.write(tmp_path, SETTINGS)
    row = wandb_run.log_eval(tmp_path, REPORT, step=250, environ=KEY)
    assert row == {'eval/step': 250, 'eval/holdout/normalized_mae_policy': .3,
                   'eval/holdout/normalized_mae_repeat_state': .5, 'eval/holdout/normalized_mae_training_mean': .9}
    assert stub[0][0] == 'init' and stub[0][1]['id'] == record['id'] and stub[0][1]['config'] is None
    assert stub[1:] == [('define_metric', 'eval/step', None), ('define_metric', 'eval/*', 'eval/step'),
                        ('log', row, None), ('finish', 0)]
    assert set(wandb_run.eval_metrics(REPORT, step=250, masked=True)) == {
        'eval/step', 'eval/holdout-masked/normalized_mae_policy', 'eval/holdout-masked/normalized_mae_repeat_state',
        'eval/holdout-masked/normalized_mae_training_mean'}
    assert wandb_run.eval_metrics(REPORT, step=0, split='train')['eval/train/normalized_mae_policy'] == .3
    assert wandb_run.eval_metrics(REPORT, step=0, split='train', masked=True)['eval/step'] == 0
    assert wandb_run.eval_metrics({'episode_mean_normalized_mae': {}}, step=1) is None


def test_log_eval_failure_is_a_warning(tmp_path, stub, capsys):
    wandb_run.write(tmp_path, SETTINGS)
    assert wandb_run.log_eval(tmp_path, REPORT, step=250, environ={}) is None      # online, no key
    assert 'W&B: not logged (RuntimeError' in capsys.readouterr().err and stub == []


def test_scalars_drop_step_strings_and_flags():
    metrics = {'step': 3, 'loss': .5, 'grad_norm': 2, 'optimizer_device': 'cpu', 'all_weights_trainable': True}
    assert wandb_run.scalars(metrics) == {'loss': .5, 'grad_norm': 2}
