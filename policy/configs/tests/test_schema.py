"""Configs load, and cloak configs produce the recipes policy.training.cloak.experiment expects."""
from pathlib import Path
import tempfile
import unittest

import pytest

from policy import configs as config
ROOT = Path(__file__).resolve().parents[3]


class Configs(unittest.TestCase):
    def test_every_config_loads(self):
        for path in sorted((ROOT / 'configs' / 'policy').glob('*.yaml')):
            with self.subTest(path.name):
                cfg = config.load(path)
                recipe = cfg.recipe()
                self.assertEqual(recipe['schema'], 'actathon.cloak.experiment.v1')
                self.assertEqual(recipe['evaluate_steps'][-1], recipe['training']['steps'])

    def test_trainer_cli_defaults_are_the_train_defaults(self):
        from policy.configs.schema import TRAIN_DEFAULTS
        from policy.training.cloak.__main__ import build_parser
        args = vars(build_parser().parse_args(['train', '--backend', 'b', '--data', 'd', '--run-dir', 'r']))
        self.assertEqual({k: args[k] for k in TRAIN_DEFAULTS}, TRAIN_DEFAULTS)

    def test_rejects_bad_configs(self):
        base = 'name: x\ntask: dirt\n'
        for text in ('name: x\nkind: dqn\n', base + 'trian: {}\n', base + 'train: {batch: 30, fsdp: 4}\nlambda: {gpus: [0,1,2,3]}\n',
                     base + 'train: {fsdp: 2}\n', base + 'reconstruct: {method: colmap}\n', base + 'reconstruct: {runner: k8s}\n', 'task: dirt\n'):
            with self.subTest(text), tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
                f.write(text)
                f.flush()
                with self.assertRaises(ValueError):
                    config.load(f.name)



# train.wandb -------------------------------------------------------------------------------------------------------

def _load(tmp_path, wandb):
    import yaml
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw.setdefault('train', {})['wandb'] = wandb
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


def test_wandb_is_off_by_default_and_not_in_the_recipe(tmp_path):
    cfg = config.load('dirt-orb')
    assert cfg.wandb == {'project': 'umi-excavator', 'entity': None, 'mode': 'disabled', 'tags': [],
                         'name': 'dirt-orb', 'group': 'dirt-orb'}
    tracked = _load(tmp_path, {'mode': 'online', 'tags': ['dirt']})
    assert tracked.recipe() == cfg.recipe()
    assert 'wandb' not in tracked.training
    assert _load(tmp_path, False).wandb['mode'] == 'disabled' and _load(tmp_path, None).wandb['mode'] == 'disabled'


def test_wandb_shorthand_and_full_mapping(tmp_path):
    assert _load(tmp_path, True).wandb == {'project': 'umi-excavator', 'entity': None, 'mode': 'online', 'tags': [],
                                           'name': 'dirt-orb', 'group': 'dirt-orb'}
    full = {'project': 'excavator', 'entity': 'team', 'mode': 'offline', 'tags': ['dirt', 'baseline'],
            'name': 'dirt-run-2', 'group': 'dirt'}
    assert _load(tmp_path, full).wandb == full
    assert _load(tmp_path, {'entity': 'team'}).wandb['mode'] == 'disabled'      # a mapping without mode stays off


@pytest.mark.parametrize('wandb, message', [
    ('yes', 'train.wandb must be true, false or a mapping'),
    ({'mode': 'cloud'}, 'train.wandb.mode'),
    ({'project': ''}, 'train.wandb.project'),
    ({'entity': 3}, 'train.wandb.entity'),
    ({'name': ''}, 'train.wandb.name'),
    ({'tags': 'dirt'}, 'train.wandb.tags'),
    ({'tags': ['']}, 'train.wandb.tags'),
    ({'tags': ['x' * 65]}, 'train.wandb.tags'),
    ({'colour': 'red'}, 'unknown train.wandb options'),
])
def test_wandb_rejects(tmp_path, wandb, message):
    with pytest.raises(ValueError, match=message.replace('.', r'\.')):
        _load(tmp_path, wandb)


# train.horizon -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('horizon', [5, 16])
def test_horizon_is_the_ten_step_chunk_the_client_runs(tmp_path, horizon):
    import yaml
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw.setdefault('train', {})['horizon'] = horizon
    path = tmp_path / 'cfg.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='train.horizon must be 10: the excavator client and hub run 10-step chunks'):
        config.load(path)
    raw['train']['horizon'] = 10
    path.write_text(yaml.safe_dump(raw))
    assert config.load(path).training['horizon'] == 10


if __name__ == '__main__':
    unittest.main()
