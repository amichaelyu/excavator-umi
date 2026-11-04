import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from policy import configs
from policy.data.cloak_dataset import digest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def frozen(tmp_path):
    """A two-episode synthetic export (train '0', holdout '1') plus reviewed masks, as in the cloak tests."""
    source, masks, output = (tmp_path / name for name in ('source', 'masks', 'prepared'))
    source.mkdir(); masks.mkdir()
    episodes, mask_entries = [], []
    contract = {'joints': ['slew', 'boom', 'stick', 'bucket'], 'units': 'radians', 'type': 'absolute next joint target',
                'provenance': 'synthetic unit-test fixture only'}
    for i, split in enumerate(('train', 'holdout')):
        n = 3
        state = np.full((n, 4), i * 100, np.float32)
        image = np.full((n, 224, 224, 3), 70, np.uint8)
        image[:, 112:, :56] = 250
        mask = np.zeros((n, 224, 224), bool); mask[:, 112:, :56] = True
        frame, time = np.arange(n) * 2, np.arange(n) / 10
        file = source / f'{i}.npz'
        np.savez_compressed(file, images=image, state=state, actions=state + np.arange(n)[:, None],
                            source_frame=frame, time_s=time)
        episodes.append({'id': str(i), 'split': split, 'frames': n, 'prompt': 'scoop the dirt',
                         'action_contract': contract, 'path': file.name, 'sha256': digest(file)})
        mfile = masks / f'{i}.npz'
        np.savez_compressed(mfile, masks=mask, source_frame=frame, time_s=time)
        mask_entries.append({'id': str(i), 'path': mfile.name, 'sha256': digest(mfile)})
    (source / 'manifest.json').write_text(json.dumps({'schema': 'actathon.pi05.real.v1', 'episodes': episodes}))
    (masks / 'manifest.json').write_text(json.dumps({'schema': 'actathon.cloak.masks.v1',
        'source_manifest_sha256': digest(source / 'manifest.json'), 'source_embodiment': 'handheld-test',
        'reviewed': True, 'provenance': 'synthetic unit-test masks', 'episodes': mask_entries}))
    return source, masks, output


class Repeat:
    """Predicts holding the current state for the whole chunk."""
    def __init__(self, horizon=2):
        self.horizon, self.seen = horizon, []

    def infer(self, obs):
        self.seen.append(obs)
        return {'actions': np.repeat(obs['state'][None], self.horizon, axis=0)}


@pytest.fixture
def make_config(tmp_path):
    """dirt-orb with its run directory and cohort under tmp_path, plus section overrides."""
    def make(**sections):
        raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
        raw['train']['run_dir'] = str(tmp_path / 'runs' / raw['name'])
        raw['cohort']['output'] = str(tmp_path / 'cohort')
        for key, value in sections.items():
            raw[key] = value
        path = tmp_path / 'cfg.yaml'
        path.write_text(yaml.safe_dump(raw))
        return configs.load(path)
    return make


def write_report(path, manifest, policy, mask_seed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'condition': 'recorded_view' if mask_seed is None else 'synthetic_edge_mask', 'mask_seed': mask_seed,
              'split': 'holdout', 'manifest_sha256': manifest, 'episodes': [{'id': '1', 'windows': 1}],
              'episode_mean_normalized_mae': {'policy': policy, 'repeat_state': .5, 'training_mean': .9}}
    path.write_text(json.dumps(report))
    return report


@pytest.fixture
def repeat():
    return Repeat


@pytest.fixture
def report_file():
    return write_report
