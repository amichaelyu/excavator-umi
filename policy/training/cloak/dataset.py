"""The model's view of a frozen CLOAK dataset frame: paired letterbox + CLOAK masking, then the π0.5 inputs.

The dataset format itself (hashes, validation, prepare, CloakDataset) is policy.data.cloak_dataset.
"""
from __future__ import annotations

import numpy as np

from . import WRIST
from .masking import prepare_view


def frame_inputs(dataset, index, *, training=False, seed=0, mode='cloak'):
    """Model inputs for ``dataset.index[index]`` (a policy.data.cloak_dataset.CloakDataset) with the next
    ``dataset.horizon`` actions; ``mode`` is the masking ablation (policy.training.cloak.masking.prepare_view)."""
    if mode == 'cloak_oov' and dataset.manifest.get('source_visibility') != 'out_of_view':
        raise ValueError('cloak_oov requires a confirmed out-of-view source')
    ep, t = dataset.index[index]
    e, a = dataset.episodes[ep]
    ids = np.minimum(np.arange(t, t + dataset.horizon), len(a['actions']) - 1)
    image, mask = prepare_view(a['images'][t], a['masks'][t], training=training,
                               rng=np.random.default_rng(seed), mode=mode,
                               content_box=dataset.manifest.get('content_box'))
    return model_inputs(image, mask, a['state'][t], e['prompt'], a['actions'][ids])


def model_inputs(image, mask, state, prompt, actions=None):
    state = np.asarray(state, np.float32)
    if state.shape != (4,) or not np.isfinite(state).all() or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Expected finite four-joint state and a nonempty prompt')
    # This is a bucket-mounted camera. Naming it base_0_rgb would bypass CLOAK
    # in the released encoder and would apply spatial image augmentation alone.
    result = {'image': {'base_0_rgb': np.zeros_like(image), WRIST: image,
                        'right_wrist_0_rgb': np.zeros_like(image)},
              'image_mask': {'base_0_rgb': np.bool_(False), WRIST: np.bool_(True),
                             'right_wrist_0_rgb': np.bool_(False)},
              'pixel_mask': {WRIST: mask, 'right_wrist_0_rgb': np.zeros_like(mask)},
              'state': state.copy(), 'prompt': prompt}
    if actions is not None:
        result['actions'] = np.asarray(actions, np.float32).copy()
    return result
