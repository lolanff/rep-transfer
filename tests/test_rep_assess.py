import jax.numpy as jnp
import pytest

from utils.functions import compute_feature_sparsity


def test_compute_feature_sparsity():
    zero_features = jnp.zeros(10)
    sparsity = compute_feature_sparsity(zero_features)
    assert sparsity == 1.0

    one_features = jnp.ones(10)
    sparsity = compute_feature_sparsity(one_features)
    assert sparsity == 0.0

    mixed_features = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    sparsity = compute_feature_sparsity(mixed_features)
    assert sparsity == 0.5

    sign_features = jnp.array([-1, 1, 0, -1])
    sparsity = compute_feature_sparsity(sign_features)
    assert sparsity == 0.25

    float_features = jnp.array([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11])
    sparsity = compute_feature_sparsity(float_features)
    assert sparsity == pytest.approx(1 / len(float_features))

    batch_features = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        ]
    )
    sparsity = compute_feature_sparsity(batch_features)
    assert sparsity == 0.75