import haiku as hk
import jax.numpy as jnp

import sys

from pathlib import Path

from representations.networks import NetworkBuilder
from utils.hk import MultiLayerHead


def test_MazeNetReLU():
    builder = NetworkBuilder(
        input_shape=(15, 15, 3),
        params={
            "hidden": 32,
            "type": "MazeNetReLU",
        },
        seed=0,
    )

    actions = 4
    feature_function = builder.getFeatureFunction()

    q_function = builder.addHead(lambda: MultiLayerHead(actions=actions, name='q'))
    
    params = builder.getParams()

    x = jnp.zeros((1, 15, 15, 3))
    phi = feature_function(params, x)
    assert phi.activations["conv"].shape == (1, 14, 14, 32)
    assert phi.activations["conv_1"].shape == (1, 8, 8, 16)
    assert phi.activations["flatten"].shape == (1, 1024)
    assert phi.activations["phi"].shape == (1, 32)
    q = q_function(params, phi.out)
    # TODO: check the shape of the activations for the head, need to switch back to accumulatingSequence to do this
    # assert phi.activations["head"].shape == (1, 64)
    # assert phi.activations["head_1"].shape == (1, 64)
    assert q.shape == (1, 4)


def MazeNetFTA():
    builder = NetworkBuilder(
        input_shape=(15, 15, 3),
        params={
            "eta": 0.2,
            "hidden": 64,
            "type": "MazeNetFTA",
        },
        seed=0,
    )

    actions = 4
    feature_function = builder.getFeatureFunction()

    q_function = builder.addHead(lambda: MultiLayerHead(width=64, actions=actions, name='q'))
    params = builder.getParams()

    x = jnp.zeros((1, 15, 15, 3))
    phi = feature_function(params, x)
    assert phi.activations["conv"].shape == (1, 14, 14, 32)
    assert phi.activations["conv_1"].shape == (1, 8, 8, 16)
    assert phi.activations["flatten"].shape == (1, 1024)
    assert phi.activations["phi"].shape == (1, 640)
    q = q_function(params, phi.out)
    # TODO: check the shape of the activations for the head, need to switch back to accumulatingSequence to do this
    # assert phi.activations["head"].shape == (1, 64)
    # assert phi.activations["head_1"].shape == (1, 64)
    assert q.shape == (1, 4)