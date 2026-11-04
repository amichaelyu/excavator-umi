import os

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('CLOAK_BACKEND'), reason='Needs the pinned CLOAK runtime')
def test_collectives_after_upstream_tensorflow_import():
    """GPU TensorFlow can export incompatible NCCL symbols into the JAX process."""
    from policy.training.cloak.backend import activate
    activate(os.environ['CLOAK_BACKEND'])
    import jax
    import jax.numpy as jnp
    count = jax.local_device_count()
    if count < 2 or jax.default_backend() != 'gpu':
        pytest.skip('Requires at least two GPUs')
    reduce = jax.pmap(lambda x: jax.lax.psum(x, 'devices'), axis_name='devices')
    result = reduce(jnp.arange(count, dtype=jnp.float32)).block_until_ready()
    np.testing.assert_array_equal(result, np.full(count, count * (count - 1) / 2))
