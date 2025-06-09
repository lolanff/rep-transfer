from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch

from algorithms.nn.DQN import DQN, AgentState, q_loss
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

class DQNAux(DQN):
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
        
        agent_positions = jnp.argwhere(state[:, :, :, 2] == 255.0, size=state.shape[0])[:, 1:]
        
        # Compare agent positions to subgoal
        terminals = jnp.all(agent_positions == subgoal, axis=1)
        
        rewards = terminals.astype(jnp.float32)
        terminals = terminals.astype(jnp.int32)
        
        return rewards, terminals

    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch, weights: jax.Array):
        # Main Q-values
        phi = self.phi(params, batch.x).out
        phi_p = self.phi(target, batch.xp).out

        if self.rep_params.get("frozen"):
            phi = jax.lax.stop_gradient(phi)

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        # Compute main loss
        batch_loss = jax.vmap(partial(q_loss, loss=self.loss), in_axes=0)
        main_losses, main_metrics = batch_loss(qs, batch.a, batch.r, batch.gamma, qsp)

        # Compute auxiliary losses for subgoals
        aux_losses = []
        if not self.rep_params.get("frozen"):
            for aux_q, subgoal in zip(self.aux_qs, self.subgoals):
                # Compute rewards and terminals for subgoals
                subgoal_rewards, subgoal_terminals = self.compute_rewards_dones(batch.xp, subgoal)
                
                aux_qs = aux_q(params, phi)
                aux_qsp = aux_q(target, phi_p)

                aux_loss, _ = batch_loss(
                    aux_qs, 
                    batch.a, 
                    subgoal_rewards, 
                    jnp.where(subgoal_terminals == 0, self.subgoal_gamma, 0), 
                    aux_qsp
                )
                aux_losses.append(aux_loss)

        # Combine all losses
        total_loss = jnp.mean(weights * main_losses)
        for aux_loss in aux_losses:
            total_loss += jnp.mean(weights * aux_loss)

        return total_loss, main_metrics