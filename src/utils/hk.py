import jax
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

from typing import Callable, Dict, Optional, Sequence

Init = hk.initializers.Initializer
Layer = Callable[[jax.Array], jax.Array]


@cxu.dataclass
class AccumulatedOutput:
    activations: Dict[str, jax.Array]
    out: jax.Array

def accumulatingSequence(fs: Sequence[Layer]):
    def _inner(x: jax.Array):
        out: Dict[str, jax.Array] = {}

        y = x
        for f in fs:
            y = f(y)
            if isinstance(f, hk.Module):
                out[f.name] = y

        return AccumulatedOutput(activations=out, out=y)
    return _inner
    
class MultiLayerHead(hk.Module):
  def __init__(self, width=64, actions=2, name=None):
    super().__init__(name=name)
    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
    b_init = hk.initializers.Constant(0)
    self.hidden1 = hk.Linear(width, w_init=w_init, b_init=b_init, name='h1')
    self.hidden2 = hk.Linear(width, w_init=w_init, b_init=b_init, name='h2')
    self.out = hk.Linear(actions, name='out')

  def __call__(self, x):
    return self.out(jax.nn.relu(self.hidden2(jax.nn.relu(self.hidden1(x)))))
