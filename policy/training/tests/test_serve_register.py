"""serve --register: only policies the hub's checkpoint registry accepts (calibrated labels) are registered, and
the refusal comes before anything starts on the box."""
import json
import logging
from pathlib import Path

import pytest
import yaml

from digital_twin.policy import checkpoints as registry
from policy import configs
from policy.training import stages as training

ROOT = Path(__file__).resolve().parents[3]


class Box:
    """A Lambda box that records Host calls."""
    address = 'ubuntu@203.0.113.7'

    def __init__(self, dry_run=False):
        self.calls, self.dry_run = [], dry_run

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append(name)


def config(tmp_path, profile):
    raw = yaml.safe_load((ROOT / 'configs/policy/dirt-orb.yaml').read_text())
    raw['train']['run_dir'] = str(tmp_path / 'run')
    raw['labels']['profile'] = profile
    path = tmp_path / f'{profile}.yaml'
    path.write_text(yaml.safe_dump(raw))
    return configs.load(path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An empty registry at configs/hub/local_checkpoints.json of a repository root in tmp_path."""
    path = tmp_path / 'configs' / 'hub' / 'local_checkpoints.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'checkpoints': []}))
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'REGISTRY', path)
    return path


@pytest.mark.parametrize('dry_run', [False, True])
def test_register_refuses_an_uncalibrated_profile_before_starting(tmp_path, hub, dry_run):
    cfg, box = config(tmp_path, 'full4'), Box(dry_run)
    before = hub.read_text()
    with pytest.raises(ValueError, match='labels.profile full4 trains handheld_reconstructed joints.*full4_calibrated'):
        training.serve(cfg, box, register=True)
    assert box.calls == [] and hub.read_text() == before


def test_register_writes_an_entry_the_registry_lists(tmp_path, hub, caplog):
    cfg, box = config(tmp_path, 'full4_calibrated'), Box()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'result.json').write_text(json.dumps({'selected_step': 500}))
    training.serve(cfg, box, register=True)
    assert box.calls == ['pull', 'ensure_image', 'docker_run']
    # An uncalibrated server's entry (what serve prints without --register) added by hand is skipped, not fatal.
    doc = json.loads(hub.read_text())
    doc['checkpoints'].insert(0, training.checkpoint_entry(config(tmp_path, 'full4'), box, 250, 8021))
    hub.write_text(json.dumps(doc))
    with caplog.at_level(logging.WARNING, logger=registry.__name__):
        rows = registry.entries(hub)
    assert [r['id'] for r in rows] == ['dirt-orb-500'] and rows[0]['policy_space'] == 'calibrated'
    assert 'dirt-orb-250' in caplog.text and 'handheld_reconstructed' in caplog.text
