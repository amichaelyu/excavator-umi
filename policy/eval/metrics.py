"""Offline action-prediction metrics: pure numpy, no model or dataset code.

An evaluation compares predicted action chunks with the recorded targets of whole held-out recordings,
next to two fixed baselines that need no model:

  policy         the model's chunk
  repeat_state   hold the current joint position for the whole chunk
  training_mean  the mean training action for every step

Per baseline and episode: ``mae_rad`` / ``rmse_rad`` (per joint, radians) and ``normalized_mae`` (absolute
error divided by the per-joint std of the training actions, averaged over steps, windows and joints).
"""
from __future__ import annotations

import numpy as np

from policy.configs.schema import EVAL_BASELINES as BASELINES

SCOPE = 'offline imitation error, not physical task success or zero-shot transfer validation'
STRATA = (('no_motion_warnings', False), ('motion_warnings', True))


def normalization(training_actions):
    """(mean, scale) of training actions [N, joints]; scale is the per-joint std, floored at 1e-6."""
    training_actions = np.asarray(training_actions)
    return training_actions.mean(0), np.maximum(training_actions.std(0), 1e-6)


def score(prediction, targets, scale):
    """Errors of ``prediction`` (broadcast to ``targets`` [windows, horizon, joints]) against ``targets``."""
    err = np.asarray(prediction) - np.asarray(targets)
    return {'mae_rad': np.abs(err).mean(axis=(0, 1)).tolist(),
            'rmse_rad': np.sqrt(np.square(err).mean(axis=(0, 1))).tolist(),
            'normalized_mae': float((np.abs(err) / scale).mean())}


def episode_mean(episodes, key='normalized_mae'):
    """Mean over episodes (each episode weighs the same, however many windows it has), per baseline."""
    return {name: float(np.mean([e[name][key] for e in episodes])) for name in BASELINES}


def motion_quality_strata(episodes):
    """``episode_mean`` separately for episodes with and without label motion warnings (present strata only)."""
    out = {}
    for label, flagged in STRATA:
        members = [e for e in episodes if bool(e.get('motion_warnings')) == flagged]
        if members:
            out[label] = {'episodes': len(members), 'episode_mean_normalized_mae': episode_mean(members)}
    return out
