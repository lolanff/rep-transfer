from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch

from algorithms.nn.DQN import DQN
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss, mse_loss
from utils.hk import MultiLayerHead

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


class DQNAux(DQN):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        # Set up the subgoals and gamma for subgoals
        # self.subgoals = np.array([[0, 0], [0, 14], [14, 0], [14, 14], [7, 7]])
        self.subgoals = np.array([[0, 0]])
        self.subgoal_gamma = 0.9
        self.state = AgentState(
            params=self.state.params,
            target_params=deepcopy(self.state.params),  # Avoid overwriting during checkpoint loading
            optim=self.state.optim,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder, num_aux_goals=1) -> None:
        # Main Q-function
        self.q = builder.addHead(
            lambda: MultiLayerHead(actions=self.actions, name='q')
        )
        # Auxiliary Q-functions for subgoals
        self.num_aux_goals = num_aux_goals
        self.aux_qs = [builder.addHead(
            lambda: MultiLayerHead(actions=self.actions, name=f'aux_q_{idx}')
        ) for idx in range(num_aux_goals)]

    def compute_rewards_dones(self, state, subgoal):
        # Un-normalizing and finding the xy coordinates of the agent
        state = (255.0 / 2.0) * (state + 1)
        
        agent_positions = jnp.argwhere(state[:, :, :, 2] == 255.0, size=state.shape[0])[:, 1:]
        
        # Compare agent positions to subgoal
        terminals = jnp.all(agent_positions == subgoal, axis=1)
        
        rewards = terminals.astype(jnp.float32)
        terminals = terminals.astype(jnp.int32)
        
        return rewards, terminals

    # Internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):  # type: ignore
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    def update(self):
        self.steps += 1

        # Only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # Skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size:
            return

        self.updates += 1

        batch = self.buffer.sample(self.batch_size)
        weights = self.buffer.isr_weights(batch.eid)
        self.state, metrics = self._computeUpdate(self.state, batch, weights)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        self.buffer.update_batch(batch, priorities=priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

        if self.updates % self.target_refresh == 0:
            self.state.target_params = self.state.params

    # Compute updates including subgoal losses
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

        return new_state, metrics

    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch, weights: jax.Array):
        # Main Q-values
        phi = self.phi(params, batch.x).out
        phi_p = self.phi(target, batch.xp).out

        if self.rep_params.get("frozen"):
            phi = jax.lax.stop_gradient(phi)

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        # Compute main loss
        main_batch_loss = jax.vmap(q_loss, in_axes=0)
        main_losses, main_metrics = main_batch_loss(qs, batch.a, batch.r, batch.gamma, qsp)

        # Compute auxiliary losses for subgoals
        aux_losses = []
        if not self.rep_params.get("frozen"):
            for aux_q, subgoal in zip(self.aux_qs, self.subgoals):
                # Compute rewards and terminals for subgoals
                subgoal_rewards, subgoal_terminals = self.compute_rewards_dones(batch.xp, subgoal)
                
                aux_qs = aux_q(params, phi)
                aux_qsp = aux_q(target, phi_p)

                aux_batch_loss = jax.vmap(q_loss, in_axes=0)
                aux_loss, _ = aux_batch_loss(aux_qs, batch.a, subgoal_rewards, jnp.where(subgoal_terminals == 0, self.subgoal_gamma, 0), aux_qsp)
                aux_losses.append(aux_loss)

        # Combine all losses
        total_loss = jnp.mean(weights * main_losses)
        for aux_loss in aux_losses:
            total_loss += jnp.mean(weights * aux_loss)

        return total_loss, main_metrics