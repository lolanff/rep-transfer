import haiku as hk
import jax.numpy as jnp
import numpy as np

import sys

from pathlib import Path

from algorithms.nn.components.RNNReplayBuffer import RNNReplayBuffer
from ReplayTables.interface import Timestep

def test_RNNReplayBuffer():
    buffer_size = 5
    sequence_length = 5
    rng = np.random.default_rng(0)
    n_step = 1
    buffer = RNNReplayBuffer(buffer_size, n_step, rng, sequence_length)
    gamma = 0.9
    for i in range(buffer_size + 3):
        buffer.add_step(Timestep(
            x=np.full((3,3,3), i),
            a=i,
            r=i*0.5,
            gamma=gamma,
            terminal=i==2 or i ==4,
            extra={'carry': np.full((1,3), i)}
        ))
    batch = buffer.sample_sequences(1)
    n_samples, *feature_dims = batch.x.shape
    n_samples = n_samples // sequence_length
    
    x = batch.x.reshape(n_samples, sequence_length, *feature_dims)
    xp = batch.xp.reshape(n_samples, sequence_length, *feature_dims)

    carry = batch.carry.reshape(n_samples, sequence_length, -1)
    initial_carry = carry[:, -1, ...]
    term = batch.terminal.reshape(n_samples, sequence_length)
    reset = jnp.hstack((jnp.zeros((term.shape[0], 1), dtype=bool), term[:, :-1]))
    print(x, xp, carry, initial_carry, reset)
    
    print(x[0,:,0,0, 0], xp[0,:,0,0, 0])
