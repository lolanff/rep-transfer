from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch, LaggedTimestep
from jax.tree_util import tree_flatten

from algorithms.nn.DRQN import DRQN, AgentState, q_loss
from algorithms.nn.NNAgent import NNAgent
from algorithms.nn.components.RNNReplayBuffer import CarryBatch
from representations.networks import NetworkBuilder
from utils.jax import huber_loss, mse_loss
from utils.hk import MultiLayerHead
from ReplayTables.interface import Timestep, TransIds
from utils.policies import egreedy_probabilities, sample

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

class DRQNAux(DRQN):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        # Set up the subgoals and gamma for subgoals before network initialization
        self.subgoals = np.array([[0, 0], [0, 14], [14, 0], [14, 14], [7, 7]])
        self.subgoal_gamma = 0.9
        super().__init__(observations, actions, params, collector, seed)

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        # Main Q-function
        self.q = builder.addHead(
            lambda: MultiLayerHead(actions=self.actions, name='q')
        )
        # Auxiliary Q-functions for subgoals
        self.num_aux_goals = len(self.subgoals)
        self.aux_qs = [
            builder.addHead(lambda idx=idx: MultiLayerHead(actions=self.actions, name=f'aux_q_{idx}'))
            for idx in range(self.num_aux_goals)
        ]

    def compute_rewards_dones(self, state, subgoal):
        # Un-normalizing and finding the xy coordinates of the agent
        state = (255.0 / 2.0) * (state + 1)
        state = state.reshape(-1, *state.shape[2:])
        
        agent_positions = jnp.argwhere(state[:, :, :, 2] == 255.0, size=state.shape[0])[:, 1:]

        # Compare agent positions to subgoal
        terminals = jnp.all(agent_positions == subgoal, axis=1)
        
        rewards = terminals.astype(jnp.float32)
        terminals = terminals.astype(jnp.int32)
        
        return rewards, terminals


    # Loss is computed for the final action in the sequence
    def _loss(self, params: hk.Params, target: hk.Params, batch: CarryBatch, weights: jax.Array):
        x = batch.x
        xp = batch.xp
        carry = batch.carry
        carryp = batch.carryp
        term = batch.terminal
        reset = batch.reset
        a = batch.a
        r = batch.r
        gamma = batch.gamma

        # Perform burn-in
        if self.burn_in_steps > 0:
            b_x, x = jnp.hsplit(x, [self.burn_in_steps])
            b_xp, xp = jnp.hsplit(xp, [self.burn_in_steps])
            _, term = jnp.hsplit(term, [self.burn_in_steps])
            b_reset, reset = jnp.hsplit(reset, [self.burn_in_steps])
            b_carry, carry = jnp.hsplit(carry, [self.burn_in_steps])
            b_carryp, carryp = jnp.hsplit(carryp, [self.burn_in_steps])
            _, a = jnp.hsplit(a, [self.burn_in_steps])
            _, r = jnp.hsplit(r, [self.burn_in_steps])
            _, gamma = jnp.hsplit(gamma, [self.burn_in_steps])
            _, weights = jnp.hsplit(weights, [self.burn_in_steps])
            
            carry = carry.at[:, 0].set(jax.lax.stop_gradient(self.phi(params, b_x, carry=b_carry, reset=b_reset, is_target=False)[1][:, -1, ...]))
            carryp = carryp.at[:, 0].set(jax.lax.stop_gradient(self.phi(target, b_xp, carry=b_carryp, reset=b_reset, is_target=True)[1][:, -1, ...]))

        phi = self.phi(params, x, carry=carry, reset=reset, is_target=False)[0]
        phi_p = self.phi(target, xp, carry=carryp, reset=reset, is_target=True)[0]

        if self.rep_params.get("frozen"):
            phi = jax.lax.stop_gradient(phi)

        if self.train_use_all_steps:
            # After the representation layer, we use all
            phi = phi.reshape(-1, phi.shape[-1])
            phi_p = phi_p.reshape(-1, phi_p.shape[-1])
            
            weights = weights.ravel()
            a = a.ravel()
            r = r.ravel()
            gamma = gamma.ravel()

        else:
            # After the representation layer, we just use the last
            phi = phi[:, -1, :]
            phi_p = phi_p[:, -1, :]
            
            weights = weights[:, -1]
            a = a[:, -1, ...]
            r = r[:, -1, ...]
            gamma = gamma[:, -1, ...]
            
        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)
            
        # Main Q-value loss
        batch_loss = jax.vmap(partial(q_loss, loss=self.loss), in_axes=0)
        main_losses, main_metrics = batch_loss(qs, a, r, gamma, qsp)
        chex.assert_equal_shape((weights, main_losses))
        # Compute auxiliary losses for subgoals
        aux_losses = []
        if not self.rep_params.get("frozen"):
            for aux_q, subgoal in zip(self.aux_qs, self.subgoals):
                subgoal_rewards, subgoal_terminals = self.compute_rewards_dones(xp, subgoal)
                
                aux_qs_vals = aux_q(params, phi)
                aux_qsp_vals = aux_q(target, phi_p)
                
                aux_loss, _ = batch_loss(
                    aux_qs_vals, a,
                    subgoal_rewards,
                    jnp.where(subgoal_terminals == 0, self.subgoal_gamma, 0),
                    aux_qsp_vals
                )
                aux_losses.append(aux_loss)
                
        # Combine all losses
        total_loss = jnp.mean(weights * main_losses)
        for aux_loss in aux_losses:
            total_loss += jnp.mean(weights * aux_loss)
            
        return total_loss, main_metrics
