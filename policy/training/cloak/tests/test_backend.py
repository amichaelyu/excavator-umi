"""Run with CLOAK_BACKEND pointing at the pinned Stanford checkout."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from policy.training.cloak.backend import activate
from policy.training.cloak import WRIST
from policy.training.cloak.masking import patch_mask


@pytest.fixture(scope='module', autouse=True)
def backend():
    root = os.environ.get('CLOAK_BACKEND')
    if not root:
        pytest.skip('Set CLOAK_BACKEND to test the real released model implementation')
    activate(root)


def test_majority_matches_released_backend():
    import jax.numpy as jnp
    from openpi.shared.masking import compute_patch_masks
    masks = np.random.default_rng(5).random((3, 224, 224)) > .5
    actual = compute_patch_masks(jnp.asarray(masks), None, 'majority')
    np.testing.assert_array_equal(actual, np.stack([patch_mask(m) for m in masks]))


@pytest.mark.parametrize('scan', [False, True])
def test_real_siglip_blocks_masked_pixels_and_gradients(scan):
    import jax
    import jax.numpy as jnp
    from openpi.models.siglip import Module
    model = Module(num_classes=8, variant=None, patch_size=(14, 14), width=32,
                   depth=2, num_heads=4, mlp_dim=64, pool_type='none', scan=scan,
                   head_zeroinit=False)
    image = jax.random.normal(jax.random.key(1), (1, 28, 28, 3))
    keep = jnp.array([[True, False, True, True]])
    params = model.init(jax.random.key(2), image, patch_mask=keep)
    modified = image.at[:, :14, 14:, :].set(100)
    a, _ = model.apply(params, image, patch_mask=keep)
    b, _ = model.apply(params, modified, patch_mask=keep)
    np.testing.assert_allclose(np.asarray(a)[:, [0, 2, 3]], np.asarray(b)[:, [0, 2, 3]], atol=1e-6)
    def loss(x):
        tokens, _ = model.apply(params, x, patch_mask=keep)
        return jnp.square(tokens[:, [0, 2, 3]]).mean()
    gradient = jax.grad(loss)(image)
    assert np.isfinite(gradient).all()
    visible_gradient = float(np.abs(gradient[:, 14:, :14, :]).max())
    assert visible_gradient > 1e-5
    # GPU convolution backward passes can leave float32 roundoff in pixels
    # whose mathematical gradient is zero. Bound it relative to a live patch.
    np.testing.assert_allclose(gradient[:, :14, 14:, :], 0, rtol=0,
                               atol=4 * np.finfo(gradient.dtype).eps * visible_gradient)


def test_released_prefix_propagates_mask_to_cross_modal_attention():
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0
    from openpi.models.model import Observation
    seen = []
    def image_encoder(image, train, patch_mask):
        seen.append(patch_mask)
        return jnp.ones((1, 256, 8)), {}
    fake = SimpleNamespace(patch_masking_strategy='majority', inject_affordance_tokens=False,
                           PaliGemma=SimpleNamespace(img=image_encoder))
    mask = jnp.zeros((1, 224, 224), bool).at[:, :14, :14].set(True)
    obs = Observation(images={WRIST: jnp.zeros((1, 224, 224, 3))},
                      image_masks={WRIST: jnp.ones((1,), bool)}, state=jnp.zeros((1, 32)),
                      pixel_masks={WRIST: mask})
    tokens, input_mask, _ = Pi0.embed_prefix(fake, obs)
    assert tokens.shape == (1, 256, 8)
    assert not bool(seen[0][0, 0]) and not bool(input_mask[0, 0])
    assert bool(input_mask[0, 1:].all())
