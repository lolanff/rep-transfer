from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch
from jax.tree_util import tree_flatten
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss, mse_loss
from utils.hk import MultiLayerHead
from utils.policies import egreedy_probabilities, sample

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

@cxu.dataclass
class AgentState:
    params: Any
    target_params: Any
    optim: optax.OptState


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    #return huber_loss(1.0, q[a], target), {
    return mse_loss(q[a], target), {
        'delta': delta,
    }

class DRQN(NNAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = params['target_refresh']
        self.carry = None
        self.state = AgentState(
            params=self.state.params,
            target_params=deepcopy(self.state.params), # without deepcopy, load_from_checkpoint overwrites params with target_params
            optim=self.state.optim,
        )
        self.cum_loss = 0
        self.running_average = 0
        self.alpha = 0.01
        self.running_average_grad = 0
        self.term = None
        self.non_zeros = 0

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        #self.q = builder.addHead(lambda: hk.Linear(self.actions, name='q'))
        self.q = builder.addHead(
            lambda: MultiLayerHead(actions=self.actions, name='q')
        )
        
    def values(self, x: np.ndarray, *args, **kwargs):
        x = np.asarray(x)

        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = np.expand_dims(x, 0)
            q, carry = self._values(self.state, x, *args, **kwargs)
            q = q[0]

        else:
            q, carry = self._values(self.state, x, *args, **kwargs)

        return jax.device_get(q), jax.device_get(carry)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        q, self.carry = self.values(obs, carry=self.carry)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi
    
    def start(self, x: np.ndarray):
        self.carry = None
        return super().start(x)

    # internal compiled version of the value function
    # TODO: carry the hidden state
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array, carry: jax.Array = None): # type: ignore
        phi = self.phi(state.params, x, carry=carry)
        return self.q(state.params, phi[0][:, -1]), phi[1][:, -1]

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size * self.sequence_length:
            return

        self.updates += 1

        batch = self.buffer.sample(self.batch_size)
        weights = self.buffer.isr_weights(batch.trans_id)
        self.state, metrics = self._computeUpdate(self.state, batch, weights)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        cur_loss = metrics['loss']
        grads = metrics['grad']
        leaves, _ = tree_flatten(grads)
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in leaves]))
        self.cum_loss += cur_loss
        self.running_average = (1 - self.alpha) * self.running_average + self.alpha * cur_loss
        self.running_average_grad = (1 - self.alpha) * self.running_average_grad + self.alpha * grad_norm
        self.term = metrics['term']
        self.buffer.update_batch(batch, priorities=priorities)
        self.non_zeros = (1 - self.alpha) * self.non_zeros + self.alpha * jnp.count_nonzero(metrics['r'])
        # for k, v in metrics.items():
        #     self.collector.collect(k, np.mean(v).item())

        if self.updates % self.target_refresh == 0:
            self.state.target_params = self.state.params   # deepcopy not needed here because optax.apply_updates produces a new pytree as params at each update, so params becomes unlinked from target_params

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            optim=optim,
        )
        metrics['grad'] = grad

        return new_state, metrics

    # Loss is computed for the final action in the sequence
    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch, weights: jax.Array):
        # Reshape the batch to have (N, T, ...)
        n_samples, *feature_dims = batch.x.shape
        n_samples = n_samples // self.sequence_length
        
        x = batch.x.reshape(n_samples, self.sequence_length, *feature_dims)
        xp = batch.xp.reshape(n_samples, self.sequence_length, *feature_dims)
        term = batch.terminal.reshape(n_samples, self.sequence_length)
        phi = self.phi(params, x, reset=term)[0]
        phi_p = self.phi(target, xp, reset=term)[0]
        if self.rep_params.get("frozen"):
            phi = jax.lax.stop_gradient(phi)

        # After the representation layer, we use all to train
        qs = self.q(params, phi)
        qs = qs.reshape(-1, qs.shape[-1])
        qsp = self.q(target, phi_p)
        qsp = qsp.reshape(-1, qsp.shape[-1])

        a = batch.a#.reshape(n_samples, self.sequence_length, 1)
        r = batch.r#.reshape(n_samples, self.sequence_length, 1)
        gamma = batch.gamma#.reshape(n_samples, self.sequence_length, 1)
        
        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, a, r, gamma, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)
        
        metrics['loss'] = loss
        metrics['term'] = term
        metrics['r'] = r

        return loss, metrics
