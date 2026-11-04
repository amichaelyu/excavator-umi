"""W&B plumbing to the Lambda box: WANDB_* from the launching shell into the container only when set (and only for
a config that tracks), values over ssh's stdin (never on a command line), wandb-run.json synced with the run."""
import json
from pathlib import Path

import pytest
import yaml

from policy import configs
from policy.eval import stage as eval_stage
from policy.training import lambda_host
from policy.training import stages as training
from policy.training.cloak import wandb_run

ROOT = Path(__file__).resolve().parents[3]
SECRET = 'wandb-secret-0123456789'


@pytest.fixture
def ssh(monkeypatch):
    """subprocess.run in lambda_host, recorded: [(argv, stdin)]."""
    calls = []

    def run(cmd, check=True, text=True, input=None, stdout=None):
        calls.append((cmd, input))
        return type('Done', (), {'stdout': '', 'returncode': 0})()

    monkeypatch.setattr(lambda_host.subprocess, 'run', run)
    return calls


@pytest.fixture
def clean_env(monkeypatch):
    for name in lambda_host.FORWARD_ENV:
        monkeypatch.delenv(name, raising=False)


def test_forwarded_env_only_has_what_is_set():
    env = {'WANDB_API_KEY': SECRET, 'WANDB_MODE': '', 'WANDB_ENTITY': 'team', 'HOME': '/Users/me'}
    assert lambda_host.forwarded_env(env) == {'WANDB_API_KEY': SECRET, 'WANDB_ENTITY': 'team'}
    assert lambda_host.forwarded_env({}) == {}


def test_docker_run_passes_names_on_the_command_and_values_on_stdin(ssh, capsys):
    host = lambda_host.Host('ubuntu@203.0.113.7')
    host.docker_run('train-x', ['python', '-m', 'policy.training.cloak.experiment'], gpus=[0],
                    env={'WANDB_API_KEY': SECRET, 'WANDB_ENTITY': 'team'})
    (cmd, stdin), = ssh
    remote = cmd[-1]
    assert remote.startswith('IFS= read -r WANDB_API_KEY; export WANDB_API_KEY; '
                             'IFS= read -r WANDB_ENTITY; export WANDB_ENTITY; mkdir -p ~/umi-excavator')
    assert ' -e WANDB_API_KEY -e WANDB_ENTITY -d ' in remote and stdin == f'{SECRET}\nteam\n'
    assert all(SECRET not in part for part in cmd) and SECRET not in capsys.readouterr().err
    ssh.clear()
    host.docker_run('train-x', ['true'], gpus=[0])
    (cmd, stdin), = ssh
    assert '-e WANDB' not in cmd[-1] and 'read -r' not in cmd[-1] and stdin is None
    with pytest.raises(ValueError, match='one-line value'):
        host.ssh('true', env={'WANDB_API_KEY': 'a\nb'})


class Box:
    """A box with nothing on it that records Host calls."""
    address, dry_run = 'ubuntu@203.0.113.7', False

    def __init__(self):
        self.calls = []

    def ssh(self, command, check=True, capture=False):
        return '' if capture else 1          # nothing running, no result.json

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))

    def train_run(self):
        return next(c for c in self.calls if c[0] == 'docker_run' and c[1][0].startswith('train-'))


def config(tmp_path, wandb=None):
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw['train']['run_dir'] = str(tmp_path / 'run')
    if wandb is not None:
        raw['train']['wandb'] = wandb
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(raw))
    return configs.load(path)


def test_train_forwards_the_key_and_syncs_the_run_id(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv('WANDB_API_KEY', SECRET)
    cfg, box = config(tmp_path, {'mode': 'online', 'tags': ['dirt']}), Box()
    training.train(cfg, box)
    assert box.train_run()[2]['env'] == {'WANDB_API_KEY': SECRET}
    record = json.loads((tmp_path / 'run' / wandb_run.FILE).read_text())
    assert record == {'id': record['id'], 'project': 'umi-excavator', 'entity': None, 'mode': 'online', 'tags': ['dirt'],
                      'name': 'dirt-orb', 'group': 'dirt-orb'}
    assert ('push_data', (cfg.run_dir,), {}) in box.calls and SECRET not in json.dumps(record)
    assert 'wandb' not in json.loads((tmp_path / 'run' / 'recipe.json').read_text())['training']
    training.train(cfg, box, resume=True)                        # a resume keeps the W&B run
    assert json.loads((tmp_path / 'run' / wandb_run.FILE).read_text())['id'] == record['id']


def test_online_without_a_key_stops_before_anything_is_synced(tmp_path, clean_env):
    box = Box()
    with pytest.raises(RuntimeError, match='WANDB_API_KEY'):
        training.train(config(tmp_path, True), box)
    assert box.calls == [] and not (tmp_path / 'run' / wandb_run.FILE).exists()


def test_offline_forwards_nothing_it_does_not_have(tmp_path, clean_env):
    box = Box()
    training.train(config(tmp_path, {'mode': 'offline'}), box)
    assert 'env' not in box.train_run()[2]


def test_a_config_without_wandb_forwards_nothing(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv('WANDB_API_KEY', SECRET)
    box = Box()
    training.train(config(tmp_path), box)
    assert 'env' not in box.train_run()[2] and not (tmp_path / 'run' / wandb_run.FILE).exists()


def test_eval_forwards_the_key_for_a_tracking_config(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv('WANDB_API_KEY', SECRET)
    box = Box()
    eval_stage._run_remote(config(tmp_path, True), box, {'gpus': [1], 'steps': [250], 'pretrained': False,
                                                          'stress': 'none', 'mask_seed': 1, 'split': 'holdout',
                                                          'predictions': False}, dry_run=False, initial=False)
    run, = [c for c in box.calls if c[0] == 'docker_run']
    assert run[1][0] == 'eval-dirt-orb' and run[2]['env'] == {'WANDB_API_KEY': SECRET}
    box = Box()
    eval_stage._run_remote(config(tmp_path), box, {'gpus': [1], 'steps': [250], 'pretrained': False,
                                                   'stress': 'none', 'mask_seed': 1, 'split': 'holdout',
                                                   'predictions': False}, dry_run=False, initial=False)
    assert 'env' not in [c for c in box.calls if c[0] == 'docker_run'][0][2]
