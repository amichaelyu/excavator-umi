import json
import os

import numpy as np
import pytest

from policy.configs.schema import TRAIN_DEFAULTS
from policy.data.cloak_dataset import CloakDataset, digest, prepare, prepare_out_of_view
from policy.training.cloak.dataset import frame_inputs
from policy.training.cloak.masking import augment, letterbox, patch_mask, prepare_view
from policy.training.cloak.policy import InferenceInputs
from policy.training.cloak import WRIST


@pytest.fixture
def frozen(tmp_path):
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


def test_prepare_preserves_labels_split_and_episode_boundaries(frozen):
    source, masks, output = frozen
    prepare(source, masks, output)
    ds = CloakDataset(output, horizon=5)
    assert len(ds) == 3
    sample = frame_inputs(ds, 2)
    np.testing.assert_array_equal(sample['actions'], np.full((5, 4), 2))
    assert ds.stats()['state'].max() == 0  # Holdout values are 100.
    assert len(CloakDataset(output, 'holdout')) == 3
    assert sample['image_mask'][WRIST] and not sample['image_mask']['base_0_rgb']
    assert sample['pixel_mask'][WRIST].any()


def test_refuses_unreviewed_masks(frozen):
    source, masks, output = frozen
    p = masks / 'manifest.json'
    data = json.loads(p.read_text()); data['reviewed'] = False; p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='reviewed'):
        prepare(source, masks, output)


def test_refuses_mask_time_shift_even_if_hash_is_valid(frozen):
    source, masks, output = frozen
    p = masks / '0.npz'
    with np.load(p) as z:
        a = dict(z)
    a['source_frame'] += 1
    np.savez_compressed(p, **a)
    meta = masks / 'manifest.json'; d = json.loads(meta.read_text())
    d['episodes'][0]['sha256'] = digest(p); meta.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='does not align'):
        prepare(source, masks, output)


def test_refuses_replaced_payload(frozen):
    source, masks, output = frozen
    prepare(source, masks, output)
    with (output / 'episode-00000.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Hash mismatch'):
        CloakDataset(output)


def test_refuses_duplicate_episode_across_splits(frozen):
    source, masks, output = frozen
    prepare(source, masks, output)
    p = output / 'manifest.json'; d = json.loads(p.read_text())
    d['episodes'][1]['id'] = d['episodes'][0]['id']; p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='unique'):
        CloakDataset(output)


def test_paired_letterbox_preserves_wide_camera_geometry():
    im = np.zeros((90, 160, 3), np.uint8)
    m = np.zeros((90, 160), bool); m[:, :80] = True; im[m] = 255
    rgb, mask = letterbox(im, m)
    assert rgb.shape == (224, 224, 3)
    assert mask[49:175, :112].all()
    assert not mask[:49].any() and not mask[175:].any() and not mask[:, 112:].any()
    assert rgb[:49].max() == 0


def test_patch_majority_is_strict_and_row_major():
    mask = np.zeros((28, 28), bool)
    mask[:7, :14] = True
    mask[:8, 14:] = True
    assert patch_mask(mask).tolist() == [False, True, False, False]


def test_training_fill_uses_original_mask_and_seed():
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    mask = np.zeros((224, 224), bool); mask[140:, :50] = True
    original = image.copy()
    a, m = augment(image, mask, np.random.default_rng(7))
    b, mb = augment(image, mask, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(m, mb)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(a[~mask], image[~mask])
    assert not np.array_equal(a[mask], image[mask])
    assert not np.array_equal(m, mask)


def test_inference_does_not_augment_or_fill():
    image = np.full((224, 224, 3), 250, np.uint8)
    mask = np.zeros((224, 224), bool); mask[100:] = True
    a, m = prepare_view(image, mask)
    np.testing.assert_array_equal(a, image); np.testing.assert_array_equal(m, mask)
    baseline, blank = prepare_view(image, mask, training=True, mode='baseline')
    np.testing.assert_array_equal(baseline, image); assert not blank.any()
    out = InferenceInputs('cloak', 4)({'rgb': image, 'embodiment_mask': mask, 'state': [0, 0, 0, 0], 'prompt': 'dig'})
    assert out['pixel_mask'][WRIST].any()
    with pytest.raises(ValueError, match='requires'):
        InferenceInputs('cloak', 4)({'rgb': image, 'state': [0, 0, 0, 0], 'prompt': 'dig'})


def test_out_of_view_augmentation_masks_content_without_filling_or_masking_padding():
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    empty = np.zeros((224, 224), bool)
    box = [0, 49, 224, 175]
    masks = []
    for seed in range(100):
        rgb, mask = prepare_view(image, empty, training=True, rng=np.random.default_rng(seed),
                                 mode='cloak_oov', content_box=box)
        np.testing.assert_array_equal(rgb, image)
        assert not mask[:49].any() and not mask[175:].any()
        assert 1 <= patch_mask(mask).sum() <= 40
        masks.append(mask)
    _, repeated = prepare_view(image, empty, training=True, rng=np.random.default_rng(7),
                               mode='cloak_oov', content_box=box)
    np.testing.assert_array_equal(masks[7], repeated)
    assert not np.array_equal(masks[0], masks[1])
    rgb, mask = prepare_view(image, empty, mode='cloak_oov')
    np.testing.assert_array_equal(rgb, image)
    assert not mask.any()
    # A target excavator can enter the image; inference must honor its true mask.
    _, mask = prepare_view(image, masks[0], mode='cloak_oov')
    np.testing.assert_array_equal(mask, masks[0])
    with pytest.raises(ValueError, match='empty source mask'):
        prepare_view(image, masks[0], training=True, rng=np.random.default_rng(0),
                     mode='cloak_oov', content_box=box)


def test_confirmed_out_of_view_preparation_preserves_real_source_contract(frozen):
    source, _, output = frozen
    evidence = {'confirmation': 'Synthetic test source is outside view',
                'source_manifest_sha256': digest(source / 'manifest.json')}
    result = prepare_out_of_view(source, output, embodiment='handheld-test', evidence=evidence,
                                content_box=[0, 0, 224, 224])
    assert result['source_visibility'] == 'out_of_view'
    ds = CloakDataset(output)
    assert not ds.episodes[0][1]['masks'].any()
    np.testing.assert_array_equal(ds.stats()['state'], 0)
    assert frame_inputs(ds, 0, training=True, mode='cloak_oov')['pixel_mask'][WRIST].any()
    assert not frame_inputs(ds, 0, mode='cloak_oov')['pixel_mask'][WRIST].any()
    assert len(CloakDataset(output, 'holdout')) == 3


def test_out_of_view_requires_evidence_for_exact_dataset(frozen):
    source, masks, output = frozen
    with pytest.raises(ValueError, match='confirmation'):
        prepare_out_of_view(source, output, embodiment='handheld-test', evidence={}, content_box=[0, 0, 224, 224])
    with pytest.raises(ValueError, match='frozen source'):
        prepare_out_of_view(source, output, embodiment='handheld-test', evidence={'confirmation': 'out of view'},
                            content_box=[0, 0, 224, 224])
    prepare(source, masks, output)
    with pytest.raises(ValueError, match='confirmed out-of-view'):
        frame_inputs(CloakDataset(output), 0, training=True, mode='cloak_oov')


def moving_joint_fixture(frozen):
    source, masks, output = frozen
    manifest = prepare(source, masks, output)
    entry = manifest['episodes'][0]
    path = output / entry['path']
    with np.load(path, allow_pickle=False) as z:
        arrays = dict(z)
    arrays['state'] = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [8, 9, 8, 9]], np.float32)
    arrays['actions'] = arrays['state'][[1, 2, 2]].copy()
    np.savez_compressed(path, **arrays)
    entry['sha256'] = digest(path)
    (output / 'manifest.json').write_text(json.dumps(manifest))
    return output, arrays


def test_delta_statistics_cover_every_future_target_and_reanchor_each_chunk(frozen):
    output, arrays = moving_joint_fixture(frozen)
    ds = CloakDataset(output, horizon=2)
    stats = ds.stats('delta')
    np.testing.assert_array_equal(stats['state'], arrays['state'])
    np.testing.assert_array_equal(stats['actions'], [[2, 2, 2, 2], [7, 7, 7, 7], [5, 5, 5, 5], [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(ds.stats()['actions'], arrays['actions'])


@pytest.mark.skipif(not os.environ.get('CLOAK_BACKEND'), reason='Needs the pinned CLOAK runtime')
@pytest.mark.parametrize('dim', [2, 4])
def test_released_delta_transforms_roundtrip_to_absolute_served_angles(frozen, tmp_path, dim):
    from policy.training.cloak.training import run, make_config, HandheldLoader
    output, arrays = moving_joint_fixture(frozen)
    if dim == 4:
        manifest = json.loads((output / 'manifest.json').read_text())
        for entry in manifest['episodes']:
            path = output / entry['path']
            with np.load(path) as z:
                a = dict(z)
            for key in ('state', 'actions'):
                a[key] = np.concatenate([a[key], a[key] * .3], axis=-1)
            entry['action_contract']['joints'] = ['slew', 'boom', 'stick', 'bucket']
            np.savez_compressed(path, **a)
            entry['sha256'] = digest(path)
            if entry['split'] == 'train':
                arrays = a
        (output / 'manifest.json').write_text(json.dumps(manifest))
    run_dir = tmp_path / 'delta-run'
    options = {**TRAIN_DEFAULTS, 'batch': 2, 'horizon': 2, 'action_representation': 'delta'}
    run(os.environ['CLOAK_BACKEND'], output, run_dir, check_data=True, **options)
    from openpi import transforms
    cfg = make_config(output, run_dir, **options)
    loader = HandheldLoader(cfg, CloakDataset(output, horizon=2), num_batches=1)
    observation, actions, _ = next(iter(loader))
    dc = loader.data_config()
    recovered = transforms.compose([transforms.Unnormalize(dc.norm_stats, use_quantiles=True),
                                    *dc.data_transforms.outputs])(
        {'state': np.asarray(observation.state).copy(), 'actions': np.asarray(actions).copy()})
    np.testing.assert_allclose(recovered['actions'][..., :dim],
                               [arrays['actions'][:2], arrays['actions'][1:]], atol=1e-5)
    assert cfg.policy_metadata['action_contract']['type'] == 'absolute next joint target'
    assert cfg.policy_metadata['action_representation'] == 'delta'


@pytest.mark.skipif(not os.environ.get('CLOAK_BACKEND'), reason='Needs the pinned CLOAK runtime')
def test_empty_source_masks_cannot_silently_become_cloak_training(frozen, tmp_path):
    from policy.training.cloak.training import run
    source, masks, output = frozen
    manifest = prepare(source, masks, output)
    entry = next(e for e in manifest['episodes'] if e['split'] == 'train')
    path = output / entry['path']
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays['masks'][:] = False
    np.savez_compressed(path, **arrays)
    entry['sha256'] = digest(path)
    (output / 'manifest.json').write_text(json.dumps(manifest))
    for mode in ('cloak', 'noaug'):
        with pytest.raises(ValueError, match='no embodiment pixels'):
            run(os.environ['CLOAK_BACKEND'], output, tmp_path / mode, check_data=True, **{**TRAIN_DEFAULTS, 'mode': mode})
    result = run(os.environ['CLOAK_BACKEND'], output, tmp_path / 'baseline',
                 check_data=True, **{**TRAIN_DEFAULTS, 'mode': 'baseline', 'batch': 2})
    assert result['trained'] is False


@pytest.mark.skipif(not os.environ.get('CLOAK_BACKEND'), reason='Needs the pinned CLOAK runtime')
def test_training_readiness_and_exact_loader_resume(frozen, tmp_path):
    from policy.training.cloak.backend import activate
    from policy.training.cloak.training import run, make_config, HandheldLoader
    source, masks, output = frozen
    prepare(source, masks, output)
    backend = activate(os.environ['CLOAK_BACKEND'])
    run_dir = tmp_path / 'run'
    # No action_representation: the record of a run made before the option existed.
    options = {**{k: v for k, v in TRAIN_DEFAULTS.items() if k != 'action_representation'}, 'batch': 2, 'horizon': 5}
    report = run(backend, output, run_dir, check_data=True, **options)
    assert report['train_frames'] == report['holdout_frames'] == 3
    assert report['actions_shape'] == [2, 5, 32]
    assert report['mask_shape'] == [2, 224, 224]
    assert not report['trained']
    # An explicit absolute (RECORD_DEFAULTS) must still accept that older run record.
    run(backend, output, run_dir, check_data=True, **options, action_representation='absolute')
    cfg = make_config(output, run_dir, **options)
    full = iter(HandheldLoader(cfg, CloakDataset(output, horizon=5), shuffle=True, num_batches=3))
    next(full); next(full)
    obs, actions, metadata = next(full)
    resumed = iter(HandheldLoader(cfg, CloakDataset(output, horizon=5), shuffle=True, skip_batches=2, num_batches=1))
    robs, ractions, rmetadata = next(resumed)
    np.testing.assert_array_equal(obs.images[WRIST], robs.images[WRIST])
    np.testing.assert_array_equal(obs.pixel_masks[WRIST], robs.pixel_masks[WRIST])
    np.testing.assert_array_equal(actions, ractions)
    np.testing.assert_array_equal(metadata['file_path'], rmetadata['file_path'])
    with pytest.raises(ValueError, match='configuration changed'):
        run(backend, output, run_dir, check_data=True, **{**options, 'batch': 3})


@pytest.mark.skipif(not os.environ.get('CLOAK_BACKEND'), reason='Needs the pinned CLOAK runtime')
@pytest.mark.parametrize('host_optimizer,microbatch', [(False, 1), (True, 1), (True, 2)])
def test_upstream_trainer_optimizer_and_checkpoint_roundtrip(frozen, tmp_path, monkeypatch, host_optimizer, microbatch):
    """A small random test model exercises the real trainer, not policy quality."""
    import dataclasses
    from policy.training.cloak.backend import activate
    activate(os.environ['CLOAK_BACKEND'])
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models.model import BaseModel, restore_params
    from openpi.models.pi0_config import Pi0Config
    from policy.training.cloak import training
    class TinyModel(BaseModel):
        def __init__(self, rng):
            super().__init__(32, 2, 64)
            self.head = nnx.Linear(32, 32, rngs=nnx.Rngs(rng))
        def compute_loss(self, rng, observation, actions, *, train=False):
            prediction = self.head(observation.state)[:, None, :]
            return jnp.square(prediction - actions).mean(-1)
        def sample_actions(self, rng, observation, **kwargs):
            return jnp.repeat(self.head(observation.state)[:, None, :], 2, axis=1)
    @dataclasses.dataclass(frozen=True)
    class TinyConfig(Pi0Config):
        def create(self, rng):
            return TinyModel(rng)
    class TinyWeights:
        def load(self, params):
            return nnx.state(TinyModel(jax.random.key(7))).to_pure_dict()
    real_config = training.make_config
    def tiny_config(*args, **kwargs):
        cfg = real_config(*args, **kwargs)
        return dataclasses.replace(cfg, model=TinyConfig(pi05=True, action_horizon=2, max_token_len=64,
                                  patch_masking_strategy='majority'), weight_loader=TinyWeights())
    monkeypatch.setattr(training, 'make_config', tiny_config)
    source, masks, output = frozen
    prepare(source, masks, output)
    run_dir = tmp_path / 'tiny-run'
    result = training.run(os.environ['CLOAK_BACKEND'], output, run_dir, steps=2, batch=2, horizon=2,
                          host_optimizer=host_optimizer, microbatch=microbatch)
    assert result['trained']
    checkpoint = run_dir / 'checkpoints/excavator_cloak/cloak/2'
    params = restore_params(checkpoint / 'params')
    assert np.isfinite(params['head']['kernel']).all()
    assert (checkpoint / 'assets/excavator_cloak/norm_stats.json').exists()
    if host_optimizer:
        before = np.array(params['head']['kernel'])
        training.run(os.environ['CLOAK_BACKEND'], output, run_dir, steps=2, batch=2, horizon=2,
                     host_optimizer=True, microbatch=microbatch, resume=True)
        after = restore_params(checkpoint / 'params')
        np.testing.assert_array_equal(before, after['head']['kernel'])
        interrupted = tmp_path / 'interrupted-run'
        partial = training.run(os.environ['CLOAK_BACKEND'], output, interrupted, steps=2, batch=2, horizon=2,
                               host_optimizer=True, microbatch=microbatch, stop_at=1)
        assert partial['completed_steps'] == 1 and not partial['training_complete']
        resumed = training.run(os.environ['CLOAK_BACKEND'], output, interrupted, steps=2, batch=2, horizon=2,
                               host_optimizer=True, microbatch=microbatch, resume=True)
        assert resumed['completed_steps'] == 2 and resumed['training_complete']
        restored = restore_params(interrupted / 'checkpoints/excavator_cloak/cloak/2/params')
        for name in params['head']:
            np.testing.assert_array_equal(params['head'][name], restored['head'][name])
