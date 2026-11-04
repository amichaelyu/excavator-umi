"""Train handheld-bucket CLOAK with the pinned upstream JAX trainer.

Only the dataset loader is adapted. SigLIP attention, downstream token gating,
flow-matching loss, optimization, sharding, and checkpointing are upstream code.
No target embodiment examples are introduced into source training.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np

from policy.configs.schema import ACTION_REPRESENTATIONS, MODES
from policy.data.cloak_dataset import CloakDataset, contract_joints, digest

from . import ASSET, UPSTREAM_REVISION, wandb_run
from .dataset import frame_inputs


# Options cloak-run.json leaves out when they equal these values, so the records of runs made before the options
# existed stay identical; make_config falls back to them when a record omits one. A record-format rule, not the
# training defaults (those are policy.configs.schema.TRAIN_DEFAULTS, filled in by the CLI and every config).
RECORD_DEFAULTS = {'action_representation': 'absolute', 'peak_lr': 5e-5, 'warmup': 25}


def write_stats(dataset, run_dir, *, action_representation='absolute'):
    from openpi.shared import normalize
    stats = {}
    for key, values in dataset.stats(action_representation).items():
        stats[key] = normalize.NormStats(mean=values.mean(0), std=values.std(0),
                                        q01=np.quantile(values, .01, axis=0),
                                        q99=np.quantile(values, .99, axis=0))
    normalize.save(Path(run_dir) / 'assets' / ASSET, stats)
    return stats


def make_config(root, run_dir, *, mode, steps, batch, fsdp, seed, horizon, save_interval, keep_period,
                resume=False, initial_checkpoint=None, host_optimizer=False, microbatch=1,
                action_representation=RECORD_DEFAULTS['action_representation'],
                peak_lr=RECORD_DEFAULTS['peak_lr'], warmup=RECORD_DEFAULTS['warmup'], tracking=None):
    """The upstream TrainConfig from a run's options (the CLI passes every one; cloak-run.json records them, minus
    RECORD_DEFAULTS). ``tracking``: the run's W&B settings (wandb_run.load) when it tracks; exp_name stays
    the mode either way, because it names the checkpoint directory (checkpoints/excavator_cloak/<mode>)."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, optimizer, weight_loaders
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if (mode not in MODES or min(steps, batch, fsdp, horizon, save_interval, keep_period, warmup) < 1
            or not 0 < peak_lr < 1 or batch % fsdp):
        raise ValueError('Invalid training options')
    if host_optimizer and (fsdp != 1 or microbatch < 1 or batch % microbatch):
        raise ValueError('CPU optimizer requires one device and a microbatch that divides the batch')
    if action_representation not in ACTION_REPRESENTATIONS:
        raise ValueError('Unknown action representation')
    manifest = json.loads((root / 'manifest.json').read_text())
    joint_order = contract_joints(manifest['episodes'][0]['action_contract'])

    @dataclasses.dataclass(frozen=True)
    class DataConfig(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            data_transforms = transforms.Group()
            if action_representation == 'delta':
                joints = transforms.make_bool_mask(len(joint_order))
                data_transforms = transforms.Group(inputs=[transforms.DeltaActions(joints)],
                                                   outputs=[transforms.AbsoluteActions(joints)])
            return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
                                       model_transforms=config.ModelTransformFactory()(model_config),
                                       data_transforms=data_transforms)

    metadata = {'schema': 'actathon.cloak.policy.v1', 'upstream_revision': UPSTREAM_REVISION,
                'mode': mode, 'source_embodiment': manifest['source_embodiment'],
                'source_visibility': manifest.get('source_visibility', 'visible'),
                'masking_recipe': 'edge_capsules_v1' if mode == 'cloak_oov' else mode,
                'action_representation': action_representation,
                'action_contract': manifest['episodes'][0]['action_contract'],
                'action_horizon': horizon, 'manifest_sha256': digest(root / 'manifest.json'),
                'task_group': manifest.get('task_group'),
                'motion_source': manifest.get('motion_source'),
                # Joint space of the labels (policy.configs.schema.LABEL_PROFILES): 'calibrated' = the real machine's ranges.
                'policy_space': manifest['episodes'][0]['action_contract'].get('policy_space'),
                'input': f'rgb uint8 HWC, embodiment_mask bool HW, state {joint_order} radians, prompt'}
    wandb = ({'wandb_enabled': True, 'project_name': tracking.get('project') or 'umi-excavator',
              'wandb_entity': tracking.get('entity')} if tracking else {'wandb_enabled': False, 'project_name': 'umi-excavator'})
    return config.TrainConfig(
        name='excavator_cloak', exp_name=mode, **wandb,
        model=Pi0Config(pi05=True, action_horizon=horizon, max_token_len=64,
                        patch_masking_strategy=None if mode == 'baseline' else 'majority'),
        data=DataConfig(repo_id=str(root), assets=config.AssetsConfig(
            assets_dir=str(run_dir / 'assets'), asset_id=ASSET)),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(initial_checkpoint).resolve() / 'params') if initial_checkpoint
            else 'gs://openpi-assets/checkpoints/pi05_base/params'),
        checkpoint_base_dir=str(run_dir / 'checkpoints'), assets_base_dir=str(run_dir / 'assets'),
        # Cosine decay to peak/10. Release 20261005: peak 5e-5 at batch 32. Retrain 20261013: 1e-4 at
        # batch 128 (sqrt scaling for 4x the batch).
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(warmup, max(1, steps // 10)),
            peak_lr=peak_lr, decay_steps=steps, decay_lr=peak_lr / 10),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=None,
        batch_size=batch, fsdp_devices=fsdp, num_workers=0, seed=seed,
        num_train_steps=steps, log_interval=10, save_interval=save_interval, keep_period=keep_period,
        resume=resume, policy_metadata=metadata)


class HandheldLoader:
    def __init__(self, cfg, dataset, *, sharding=None, shuffle=False, num_batches=None,
                 skip_batches=0, skip_norm_stats=False):
        import jax
        from openpi import transforms
        self.cfg, self.dataset, self.sharding = cfg, dataset, sharding
        self.shuffle, self.num_batches, self.skip_batches = shuffle, num_batches, skip_batches
        self.dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        if not skip_norm_stats and self.dc.norm_stats is None:
            raise ValueError('Training-only normalization assets are missing')
        self.transform = transforms.compose([
            *self.dc.data_transforms.inputs,
            transforms.Normalize({} if skip_norm_stats else self.dc.norm_stats,
                                 use_quantiles=self.dc.use_quantile_norm),
            *self.dc.model_transforms.inputs])
        if jax.process_count() != 1:
            raise ValueError('The handheld loader currently supports one host')

    def data_config(self):
        return self.dc

    def __iter__(self):
        import jax
        from openpi.models.model import Observation
        n, batch = len(self.dataset), self.cfg.batch_size
        cached_epoch, order = None, None
        step, emitted = self.skip_batches, 0
        while self.num_batches is None or emitted < self.num_batches:
            samples, episode_ids = [], []
            for slot in range(batch):
                epoch, offset = divmod(step * batch + slot, n)
                if epoch != cached_epoch:
                    rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, epoch]))
                    order = rng.permutation(n) if self.shuffle else np.arange(n)
                    cached_epoch = epoch
                idx = int(order[offset])
                # Batch number and slot reproduce augmentation exactly after resume.
                seed = np.random.SeedSequence([self.cfg.seed, step, slot])
                sample = frame_inputs(self.dataset, idx, training=True, seed=seed,
                                      mode=self.cfg.policy_metadata['mode'])
                transformed = self.transform(sample)
                transformed.pop('prompt', None)  # Text is tokenized; JAX accepts numeric leaves only.
                samples.append(transformed)
                ep, _ = self.dataset.index[idx]
                episode_ids.append(self.dataset.episodes[ep][0]['id'])
            packed = jax.tree.map(lambda *xs: np.stack(xs), *samples)
            def put(x):
                return jax.device_put(x) if self.sharding is None else jax.make_array_from_process_local_data(self.sharding, x)
            observation = Observation.from_dict(jax.tree.map(put, packed))
            yield observation, put(packed['actions']), {'file_path': np.asarray(episode_ids)}
            step += 1
            emitted += 1


def run(backend, root, run_dir, *, check_data=False, stop_at=None, **options):
    from .backend import activate
    # W&B (<run_dir>/wandb-run.json, written by `python -m policy train` when train.wandb is on): stop now, in the
    # data check that starts every experiment, not after the JAX compile, if the run could not log.
    tracking = wandb_run.load(run_dir)
    if wandb_run.check(tracking) == 'disabled':
        tracking = None
    backend = activate(backend)
    if stop_at is not None and (stop_at < 1 or not options.get('host_optimizer')):
        raise ValueError('stop_at requires the CPU optimizer and a positive step')
    dataset = CloakDataset(root, horizon=options['horizon'])
    mode = options['mode']
    if mode == 'cloak_oov' and dataset.manifest.get('source_visibility') != 'out_of_view':
        raise ValueError('cloak_oov requires a confirmed out-of-view source')
    if mode in ('cloak', 'noaug') and not any(a['masks'].any() for _, a in dataset.episodes):
        raise ValueError('Training masks contain no embodiment pixels; resolve the source mask before CLOAK training')
    # Validate the holdout independently before launching any optimizer steps.
    holdout = CloakDataset(root, split='holdout', horizon=dataset.horizon)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {'upstream_revision': UPSTREAM_REVISION,
              'manifest_sha256': digest(Path(root) / 'manifest.json'),
              # Defaults are omitted so records of earlier runs stay identical.
              'options': {k: str(v) if isinstance(v, Path) else v for k, v in options.items()
                          if k != 'resume' and not (k in RECORD_DEFAULTS and v == RECORD_DEFAULTS[k])}}
    record_path = run_dir / 'cloak-run.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise ValueError('Run configuration changed; use a separate run directory')
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    stats_path = run_dir / 'assets' / ASSET / 'norm_stats.json'
    if not stats_path.exists():
        write_stats(dataset, run_dir, action_representation=options.get(
            'action_representation', RECORD_DEFAULTS['action_representation']))
    cfg = make_config(root, run_dir, tracking=tracking, **options)
    def factory(config, **kwargs):
        if config != cfg:
            raise ValueError('Unexpected configuration in handheld loader')
        return HandheldLoader(config, dataset, **kwargs)
    observation, actions, _ = next(iter(factory(cfg, num_batches=1)))
    readiness = {'train_episodes': len(dataset.episodes), 'holdout_episodes': len(holdout.episodes),
                 'train_frames': len(dataset), 'holdout_frames': len(holdout),
                 'actions_shape': list(actions.shape),
                 'mask_shape': list(observation.pixel_masks['left_wrist_0_rgb'].shape),
                 'trained': False}
    (run_dir / 'readiness.json').write_text(json.dumps(readiness, indent=2) + '\n')
    if check_data:
        return readiness
    from openpi.training import data_loader
    spec = importlib.util.spec_from_file_location('umi_excavator_cloak_upstream_train', backend / 'src/train.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_data_loader
    recipe = {**{k: str(v) if isinstance(v, Path) else v for k, v in options.items() if k != 'resume'},
              'manifest_sha256': record['manifest_sha256'], 'upstream_revision': UPSTREAM_REVISION,
              'engine': 'host_optimizer' if options.get('host_optimizer') else 'upstream',
              'train_episodes': len(dataset.episodes), 'holdout_episodes': len(holdout.episodes)}
    with wandb_run.session(run_dir, config=recipe) as tracker:
        if tracker is not None:
            # This process's run is already initialized (stable id, group, tags, dir). Upstream's init_wandb would
            # start another run named after the mode; its wandb.log calls (loss, grad norm, GPU) land in this one.
            trainer.init_wandb = lambda *args, **kwargs: None
        try:
            data_loader.create_data_loader = factory
            if options.get('host_optimizer'):
                from .host_training import main
                completed_steps = main(trainer, cfg, factory, microbatch=options.get('microbatch', 1),
                                       stop_at=stop_at, tracker=tracker)
            else:
                trainer.main(cfg)
                completed_steps = cfg.num_train_steps
        finally:
            data_loader.create_data_loader = original
    result = {**readiness, 'trained': True, 'completed_steps': completed_steps,
              'training_complete': completed_steps >= cfg.num_train_steps}
    (run_dir / 'readiness.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
