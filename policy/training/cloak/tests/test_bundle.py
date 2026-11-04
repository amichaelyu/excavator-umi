import json
import shutil

import pytest

from policy.training.cloak import ASSET, UPSTREAM_REVISION
from policy.training.cloak.bundle import export_bundle, verify_bundle
from policy.data.cloak_dataset import digest


def test_bundle_survives_relocation_without_training_data_or_optimizer(tmp_path):
    source = tmp_path / 'source'
    data, run = source / 'data', source / 'run'
    data.mkdir(parents=True); run.mkdir()
    (data / 'manifest.json').write_text(json.dumps({'episodes': [{'action_contract': {
        'joints': ['slew', 'boom', 'stick', 'bucket']}}], 'task_group': 'dirt'}))
    (data / 'recording.npz').write_bytes(b'private training recording')
    (run / 'cloak-run.json').write_text(json.dumps({'upstream_revision': UPSTREAM_REVISION,
        'manifest_sha256': digest(data / 'manifest.json'), 'options': {
            'mode': 'cloak_oov', 'horizon': 10, 'initial_checkpoint': '/old/machine/initial'}}))
    checkpoint = run / 'checkpoints/excavator_cloak/cloak_oov/200'
    (checkpoint / 'params').mkdir(parents=True)
    (checkpoint / 'params/_METADATA').write_text('{}')
    (checkpoint / 'params/weights').write_bytes(b'model parameters')
    (checkpoint / '_CHECKPOINT_METADATA').write_text('{}')
    (checkpoint / 'optimizer').write_bytes(b'large optimizer state')
    for base in (run, checkpoint):
        path = base / 'assets' / ASSET
        path.mkdir(parents=True)
        (path / 'norm_stats.json').write_text('{"norm_stats":{}}')
    output = tmp_path / 'export'
    exported = export_bundle(data, run, checkpoint, output)
    relocated = tmp_path / 'someone/model'
    relocated.parent.mkdir()
    output.rename(relocated)
    shutil.rmtree(source)
    assert verify_bundle(relocated) == exported
    assert not list(relocated.rglob('*.npz')) and not list(relocated.rglob('optimizer'))
    record = json.loads((relocated / 'run/cloak-run.json').read_text())
    assert record['options']['initial_checkpoint'] is None
    assert digest(relocated / 'data/manifest.json') == record['manifest_sha256']
    params = relocated / exported['checkpoint'] / 'params/weights'
    assert params.read_bytes() == b'model parameters'
    params.write_bytes(b'corrupted payload')
    with pytest.raises(ValueError, match='payload mismatch'):
        verify_bundle(relocated)
