"""Evaluate CLOAK / π0.5 checkpoints on the whole-recording holdout of their frozen cohort.

Two levels, both run inside the training image (JAX, the pinned CLOAK backend at /opt/cloak):

  evaluate(policy, root, ...)        one loaded policy over a dataset split -> report dict
  evaluate_checkpoint(...)           load a checkpoint (or pi05_base) of a run, write one eval-*.json

Several reports of one run directory, each in a fresh ``python -m policy.eval checkpoint`` process, reusing cached
reports, and the best-step selection are policy.training.cloak.evaluator (RunEvaluator / evaluate_run): the
training experiment uses them too, and policy/ only imports downward.

Report files live in the run directory: ``eval-initial.json`` (pi05_base before fine-tuning),
``eval-<step>.json``, ``eval-<step>-masked.json`` (synthetic edge masks) and, for ``split: train``,
the same names with a ``-train`` suffix. These scores are not robot task success. A run that tracks with W&B
(``<run_dir>/wandb-run.json``) also gets each new report's scores in its W&B run
(policy.training.cloak.wandb_run.log_eval).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.configs.schema import EVALUATE_SPLITS as SPLITS
from policy.data.cloak_dataset import CloakDataset, digest

from . import metrics
from .report import parse_name


def evaluate(policy, root, *, horizon=10, progress=None, mask_seed=None, reference=None,
             predictions_path=None, split='holdout'):
    """Score ``policy.infer`` on every complete, non-overlapping action chunk of the ``split`` recordings.

    reference: dataset whose train split sets the normalization scale and training-mean baseline
    (default root; pass the policy's own training data when root is a foreign evaluation set).
    mask_seed: replace each frame's mask with reproducible synthetic edge occlusions (out-of-view sources only).
    predictions_path: optional .npz for the per-window chunks ('<id>/policy|target' [windows,horizon,joints],
    '<id>/state|time_s' at each window start)."""
    from policy.training.cloak.masking import augment_out_of_view
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    train = CloakDataset(reference or root, horizon=horizon)
    held = CloakDataset(root, split=split, horizon=horizon)
    dim = len(held.joints)
    if mask_seed is not None and held.manifest.get('source_visibility') != 'out_of_view':
        raise ValueError('Synthetic edge-mask evaluation requires an out-of-view source')
    mean, scale = metrics.normalization(train.stats()['actions'])
    episodes, skipped, saved = [], [], {}
    for episode_index, (entry, a) in enumerate(held.episodes):
        predictions, targets, repeats = [], [], []
        for t in range(0, len(a['actions']) - horizon + 1, horizon):
            mask = a['masks'][t].copy()
            if mask_seed is not None:
                _, mask = augment_out_of_view(a['images'][t], mask,
                    np.random.default_rng(np.random.SeedSequence([mask_seed, episode_index, t])),
                    held.manifest['content_box'])
            result = policy.infer({'rgb': a['images'][t].copy(), 'embodiment_mask': mask,
                                   'state': a['state'][t].copy(), 'prompt': entry['prompt']})
            action = np.asarray(result['actions'])
            if action.shape != (horizon, dim) or not np.isfinite(action).all():
                raise ValueError(f'Expected finite policy actions [horizon,{dim}]')
            predictions.append(action)
            targets.append(a['actions'][t:t + horizon])
            repeats.append(np.broadcast_to(a['state'][t], (horizon, dim)))
        if not predictions:
            skipped.append(entry['id'])
            continue
        targets = np.asarray(targets)
        if predictions_path is not None:
            saved.update({f'{entry["id"]}/policy': np.asarray(predictions, np.float32),
                          f'{entry["id"]}/target': targets.astype(np.float32),
                          f'{entry["id"]}/state': np.asarray(a['state'][:len(targets) * horizon:horizon], np.float32),
                          f'{entry["id"]}/time_s': np.asarray(a['time_s'][:len(targets) * horizon:horizon], np.float64)})
        episodes.append({'id': entry['id'], 'windows': len(targets),
                         'policy': metrics.score(predictions, targets, scale),
                         'group': entry.get('group'),
                         'motion_warnings': entry.get('motion_warnings', []),
                         'repeat_state': metrics.score(repeats, targets, scale),
                         'training_mean': metrics.score(mean, targets, scale)})
        if progress:
            progress(episodes[-1])
    if not episodes:
        raise ValueError(f'No {"held-out" if split == "holdout" else split} episode is long enough for a complete action chunk')
    if predictions_path is not None:
        np.savez_compressed(predictions_path, **saved)
    return {'scope': metrics.SCOPE,
            'condition': 'recorded_view' if mask_seed is None else 'synthetic_edge_mask', 'mask_seed': mask_seed,
            'split': split,
            'manifest_sha256': digest(held.root / 'manifest.json'), 'horizon': horizon,
            **({'normalization_manifest_sha256': digest(train.root / 'manifest.json')} if reference else {}),
            'policy_metadata': getattr(policy, 'metadata', {}),
            'joint_order': held.joints, 'training_action_std': scale.tolist(),
            'episodes': episodes, 'skipped_short_episodes': skipped,
            'motion_quality_strata': metrics.motion_quality_strata(episodes),
            'episode_mean_normalized_mae': metrics.episode_mean(episodes)}


def evaluate_checkpoint(backend, data, run_dir, output, *, checkpoint=None, pretrained=False, mask_seed=None,
                        policy_data=None, predictions=None, split='holdout', progress=None):
    """Load a trained ``checkpoint`` of ``run_dir`` (or, with ``pretrained``, the pi05_base initialization
    with the run's normalization) and write its report to ``output``. ``policy_data``: the dataset the run
    trained on, when ``data`` is a different evaluation set."""
    from policy.training.cloak.policy import load_policy
    policy = load_policy(backend, policy_data or data, run_dir, checkpoint, pretrained=pretrained)
    result = evaluate(policy, data, horizon=policy.metadata['action_horizon'], mask_seed=mask_seed,
                      progress=progress or (lambda e: print(json.dumps(e), flush=True)),
                      reference=policy_data, predictions_path=predictions, split=split)
    if policy_data:
        result['evaluation_data'] = {'data': str(data), 'policy_data': str(policy_data)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    parsed = parse_name(output.name)
    if parsed and not policy_data and output.parent.resolve() == Path(run_dir).resolve():
        from policy.training.cloak import wandb_run
        wandb_run.log_eval(run_dir, result, step=parsed['step'], masked=parsed['masked'], split=parsed['split'])
    return result
