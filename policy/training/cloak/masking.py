"""CLOAK sections 4.1 and B.1, at the policy's 224-pixel resolution.

True pixels identify the embodiment. True patch values mean DROP, whereas
the upstream SigLIP patch_mask uses True for KEEP. External views are untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from policy.data.cloak_dataset import checked_content_box


def validate_pair(image, mask):
    image, mask = np.asarray(image), np.asarray(mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected uint8 RGB image in HWC order')
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2]:
        raise ValueError('Expected boolean embodiment mask aligned with RGB pixels')
    if min(mask.shape) < 1:
        raise ValueError('Empty image')
    return image, mask


def _letterbox_box(shape, size):
    """``(nh, nw, y, x)``: the aspect-preserving, centred placement of an ``(h, w)`` frame in ``size``x``size``."""
    h, w = shape
    ratio = max(w / size, h / size)
    nh, nw = max(1, int(h / ratio)), max(1, int(w / ratio))
    return nh, nw, (size - nh) // 2, (size - nw) // 2


def letterbox_mask(mask, size=224):
    """The mask half of :func:`letterbox` alone (nearest-neighbour, never stretched)."""
    nh, nw, y, x = _letterbox_box(mask.shape, size)
    out = np.zeros((size, size), bool)
    out[y:y + nh, x:x + nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    return out


def letterbox(image, mask, size=224):
    """Resize RGB and silhouette with identical geometry; never stretch a mask."""
    image, mask = validate_pair(image, mask)
    nh, nw, y, x = _letterbox_box(mask.shape, size)
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[y:y + nh, x:x + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return rgb, letterbox_mask(mask, size)


def patch_mask(mask, patch_size=14):
    """Equation 4: strict majority, including exactly-half patches as visible."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or patch_size < 1:
        raise ValueError('Expected 2D boolean mask and positive patch size')
    h, w = mask.shape
    if h % patch_size or w % patch_size:
        raise ValueError('Image dimensions must be divisible by patch size')
    return (mask.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
            .mean(axis=(1, 3)) > 0.5).reshape(-1)


def augment(image, mask, rng):
    """Roll-and-fill, add 1-2 boundary capsules, remove 1-2 interior disks.

    Constants are paper pixels at 224x224. Inputs are never mutated. The fill
    uses the original mask, before disk removal can reveal embodiment pixels.
    """
    image, mask = validate_pair(image, mask)
    if mask.shape != (224, 224):
        raise ValueError('Apply paired letterboxing to 224x224 before augmentation')
    h, w = mask.shape
    shift = tuple(int(rng.integers(n // 3, 2 * n // 3 + 1)) for n in (h, w))
    filled = np.where(mask[..., None], np.roll(image, shift, axis=(0, 1)), image)
    m = mask.astype(np.uint8)
    boundary = np.argwhere(m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)))
    for _ in range(int(rng.integers(1, 3))):
        if not len(boundary):
            break
        y, x = boundary[rng.integers(len(boundary))]
        radius, length = int(rng.integers(12, 17)), int(rng.integers(32, 65))
        theta = np.deg2rad(rng.uniform(-30, 30))
        sign = rng.choice([-1, 1])
        end = (int(x + length * np.sin(theta)), int(y + sign * length * np.cos(theta)))
        cv2.line(m, (int(x), int(y)), end, 1, thickness=2 * radius)
    inside = np.argwhere(m)
    for _ in range(int(rng.integers(1, 3))):
        if not len(inside):
            break
        y, x = inside[rng.integers(len(inside))]
        cv2.circle(m, (int(x), int(y)), int(rng.integers(8, 17)), 0, thickness=-1)
    return filled, m.astype(bool)


def augment_out_of_view(image, mask, rng, content_box):
    """Excavator extension: edge-attached occlusions for an invisible source tool.

    There are no source-tool pixels to roll-fill. Keep RGB intact and seed the
    paper's capsule/disk perturbations on the image-content boundary instead
    of a nonexistent silhouette. Padding is excluded. At most a quarter of
    content-overlapping tokens may be removed, and at least one must be removed.
    This is an explicit adaptation, not the paper's visible-source recipe.
    """
    image, mask = validate_pair(image, mask)
    if mask.shape != (224, 224) or mask.any():
        raise ValueError('Out-of-view augmentation needs a 224x224 empty source mask')
    x0, y0, x1, y1 = checked_content_box(content_box)
    content = np.zeros_like(mask)
    content[y0:y1, x0:x1] = True
    content_tokens = content.reshape(16, 14, 16, 14).any(axis=(1, 3)).sum()
    for _ in range(32):
        m = np.zeros(mask.shape, np.uint8)
        for _ in range(int(rng.integers(1, 3))):
            side = int(rng.integers(4))
            if side < 2:
                start = (x0 if side == 0 else x1 - 1, int(rng.integers(y0, y1)))
                angle = 0 if side == 0 else np.pi
            else:
                start = (int(rng.integers(x0, x1)), y0 if side == 2 else y1 - 1)
                angle = np.pi / 2 if side == 2 else -np.pi / 2
            angle += np.deg2rad(rng.uniform(-30, 30))
            length, radius = int(rng.integers(32, 65)), int(rng.integers(12, 17))
            end = (round(start[0] + length * np.cos(angle)), round(start[1] + length * np.sin(angle)))
            cv2.line(m, start, end, 1, thickness=2 * radius)
            cv2.circle(m, start, radius, 1, thickness=-1)
            cv2.circle(m, end, radius, 1, thickness=-1)
        m[~content] = 0
        for _ in range(int(rng.integers(1, 3))):
            inside = np.argwhere(m)
            if not len(inside):
                break
            y, x = inside[rng.integers(len(inside))]
            cv2.circle(m, (int(x), int(y)), int(rng.integers(8, 17)), 0, thickness=-1)
        masked = m.astype(bool)
        dropped = patch_mask(masked).sum()
        if 1 <= dropped <= max(1, int(content_tokens * .25)):
            return image.copy(), masked
    raise ValueError('content_box is too small for out-of-view capsule augmentation')


def prepare_view(image, mask, *, training=False, rng=None, mode='cloak', content_box=None):
    """Ablations share geometry: baseline, noaug, or the full CLOAK recipe."""
    if mode not in ('cloak', 'cloak_oov', 'noaug', 'baseline'):
        raise ValueError('Unknown masking mode')
    image, mask = letterbox(image, mask)
    if mode == 'baseline':
        return image, np.zeros_like(mask)
    if training and mode in ('cloak', 'cloak_oov'):
        if rng is None:
            raise ValueError('Training augmentation needs an explicit random generator')
        if mode == 'cloak_oov':
            return augment_out_of_view(image, mask, rng, content_box)
        return augment(image, mask, rng)
    # Paper inference uses the constructed mask and original RGB, without fill.
    return image, mask
