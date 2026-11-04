"""CLOAK inference on raw bucket RGB with geometric masks, optionally retargeted.

This module produces action chunks only. It never sends motor commands.
Existing unmasked checkpoints are not silently treated as CLOAK checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ASSET, UPSTREAM_REVISION
from .dataset import model_inputs
from .masking import prepare_view


class InferenceInputs:
    def __init__(self, mode='cloak', action_dim=2):
        self.mode, self.action_dim = mode, action_dim

    def __call__(self, data):
        if 'embodiment_mask' not in data:
            raise ValueError('CLOAK requires a geometric embodiment_mask for this exact RGB frame')
        if np.asarray(data['state']).shape != (self.action_dim,):
            raise ValueError(f'This policy requires {self.action_dim} state joints in its declared order')
        image, mask = prepare_view(data['rgb'], data['embodiment_mask'], mode=self.mode)
        return model_inputs(image, mask, data['state'], data['prompt'])


class BatchInputs:
    def __call__(self, data):
        import jax
        data = {k: v for k, v in data.items() if k != 'prompt'}
        return jax.tree.map(lambda x: np.asarray(x)[None], data)


class ActionOutputs:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim

    def __call__(self, data):
        actions = np.asarray(data['actions'])[..., :self.action_dim]
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim or not np.isfinite(actions).all():
            raise ValueError('Policy returned invalid bucket actions')
        return {**data, 'actions': actions}


def load_policy(backend, root, run_dir, checkpoint=None, *, pretrained=False):
    from .backend import activate
    activate(backend)
    from openpi import transforms
    from openpi.models import model
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from .training import make_config
    from policy.data.cloak_dataset import digest
    run_dir = Path(run_dir)
    record = json.loads((run_dir / 'cloak-run.json').read_text())
    if record['upstream_revision'] != UPSTREAM_REVISION or record['manifest_sha256'] != digest(Path(root) / 'manifest.json'):
        raise ValueError('Run does not match the pinned backend and frozen dataset')
    cfg = make_config(root, run_dir, **record['options'])
    import jax
    jax.config.update('jax_compilation_cache_dir', str(cfg.checkpoint_dir.parent / '.jax-cache'))
    if pretrained:
        if checkpoint is not None:
            raise ValueError('Choose pretrained initialization or a trained checkpoint')
        from openpi.shared import download
        params_path = download.maybe_download(cfg.weight_loader.params_path)
        assets_path = run_dir / 'assets' / ASSET
    else:
        if checkpoint is None:
            raise ValueError('A checkpoint is required')
        checkpoint = Path(checkpoint)
        # Keep checkpoint identity bound to this run, including its masking mode.
        if not checkpoint.resolve().is_relative_to(cfg.checkpoint_dir.resolve()):
            raise ValueError('Checkpoint must belong to this CLOAK run directory')
        params_path = checkpoint / 'params'
        assets_path = checkpoint / 'assets' / ASSET
        if not params_path.is_dir():
            raise ValueError('Checkpoint params are missing')
    net = cfg.model.load(model.restore_params(params_path))
    stats = normalize.load(assets_path)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    dim = len(cfg.policy_metadata['action_contract']['joints'])
    return Policy(net, transforms=[InferenceInputs(cfg.policy_metadata['mode'], dim),
                  *dc.data_transforms.inputs,
                  transforms.Normalize(stats, use_quantiles=True),
                  *dc.model_transforms.inputs, BatchInputs()],
                  output_transforms=[transforms.Unnormalize(stats, use_quantiles=True),
                                     *dc.data_transforms.outputs, ActionOutputs(dim)],
                  sample_kwargs={'num_steps': 10}, metadata={**cfg.policy_metadata,
                      'weights': 'pretrained_initialization' if pretrained else str(checkpoint)})


class GeometryPolicy:
    """Render using target joints before converting state to source coordinates.

    geometry_state is explicit because the repository has both prior and
    calibrated joint coordinate systems. An RGB frame must be the native,
    uncropped calibrated camera frame; preprocessing happens after rendering.
    """
    def __init__(self, policy, config, retargeter=None, source_seed=None):
        from .geometry import checked_config
        from digital_twin.sim.self_mask import SelfMasker
        self.cfg = checked_config(config)
        from policy.data.cloak_dataset import JOINT_ORDERS
        if self.cfg.get('state_joints') not in JOINT_ORDERS:
            raise ValueError('Geometry state_joints must declare a supported joint order')
        if retargeter is None:
            if self.cfg['source_embodiment'] != policy.metadata['source_embodiment']:
                raise ValueError('Different source and target embodiments require retargeting')
            if (self.cfg.get('state_contract') != policy.metadata['action_contract']
                    or self.cfg['state_joints'] != policy.metadata['action_contract']['joints']):
                raise ValueError('Geometry state and policy state contracts must match')
        else:
            if (self.cfg['state_joints'] != ['boom', 'bucket']
                    or policy.metadata['action_contract']['joints'] != ['boom', 'bucket']):
                raise ValueError('The legacy cross-embodiment retargeter supports two joints only')
            if self.cfg['profile'] != retargeter.target.profile:
                raise ValueError('Renderer must use the target retargeting profile')
            if source_seed is None:
                raise ValueError('Cross-embodiment inference requires an explicit source seed')
        self.policy, self.retargeter = policy, retargeter
        self.source_seed = None if source_seed is None else np.asarray(source_seed, float)
        self.masker = SelfMasker(config=self.cfg)

    @property
    def metadata(self):
        return {**self.policy.metadata, 'geometry_state_required': True,
                'target_embodiment': self.cfg['source_embodiment']}

    def infer(self, obs):
        rgb = np.asarray(obs['rgb'])
        w, h = self.cfg['intrinsics']['image_size']
        if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
            raise ValueError('Send native calibrated RGB; do not crop, resize or blank it first')
        target_q = np.asarray(obs['geometry_state'], float)
        if target_q.shape != (len(self.cfg['state_joints']),) or not np.isfinite(target_q).all():
            raise ValueError('geometry_state must match the declared joints in radians')
        mask = self.masker.mask(dict(zip(self.cfg['state_joints'], target_q)), rgb.shape)
        source_q = target_q
        if self.retargeter is not None:
            source_q, _ = self.retargeter.target_state(target_q, self.source_seed)
            self.source_seed = source_q
        result = self.policy.infer({'rgb': rgb, 'embodiment_mask': mask,
                                    'state': source_q, 'prompt': obs['prompt']})
        if self.retargeter is not None:
            actions, residuals = self.retargeter.source_actions(result['actions'], target_q)
            result = {**result, 'actions': actions, 'retargeting': residuals}
        return result

    def close(self):
        self.masker.close()
