from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple
from ml_instrumentation.Collector import Collector
from ReplayTables.ReplayBuffer import Batch

from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.jax import huber_loss, mse_loss
from ReplayTables.interface import Timestep, TransIds
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


def q_loss(q, a, r, gamma, qp, loss='mse'):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    if loss == 'mse':
        return mse_loss(q[a], target), {
            'delta': delta,
        }
    elif loss == 'huber':    
        return huber_loss(1.0, q[a], target), {
            'delta': delta,
        }
    else:
        raise NotImplementedError

class ARDQN(NNAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        observations = observations[1:]
        super().__init__(observations, actions, params, collector, seed)
        self.num_actions = actions
        # set up the target network parameters
        self.target_refresh = params['target_refresh']
        self.loss = params.get('loss', 'mse')
        self.state = AgentState(
            params=self.state.params,
            target_params=deepcopy(self.state.params), # without deepcopy, load_from_checkpoint overwrites params with target_params
            optim=self.state.optim,
        )

    def get_feature_function(self, builder: NetworkBuilder):
        return builder.getActionRewardFeatureFunction()
    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        if self.head == "MultiLayerHead":
            self.q = builder.addHead(
                lambda: MultiLayerHead(actions=self.actions, name='q')
            )
        else:
            self.q = builder.addHead(lambda: hk.Linear(self.actions, name='q', w_init=hk.initializers.Orthogonal(np.sqrt(2))))
        

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array, a, r): # type: ignore
        phi = self.phi(state.params, x, a, r)
        return self.q(state.params, phi)

    def update(self):
        self.steps += 1

        self.update_epsilon()

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size:
            return

        self.updates += 1

        batch = self.buffer.sample_sequences(self.batch_size)
        weights_shape = batch.trans_id.shape
        weights = self.buffer.isr_weights(TransIds(np.array(batch.trans_id).ravel())).reshape(weights_shape)
        self.state, metrics = self._computeUpdate(self.state, batch, weights)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        self.buffer.update_batch(batch, priorities=priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

        if self.updates % self.target_refresh == 0:
            self.state.target_params = self.state.params   # deepcopy not needed here because optax.apply_updates produces a new pytree as params at each update, so params becomes unlinked from target_params

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch, weights: jax.Array):
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

    def _loss(self, params: hk.Params, target: hk.Params, batch, weights: jax.Array):
        x = batch.x[:, 0]
        xp = batch.xp[:, 0]
        a = batch.a[:, 0]
        action_encoded = self.encode_action(a)
        r = batch.r[:, 0]
        gamma = batch.gamma[:, 0]
        last_action_encoded = batch.last_action_encoded[:, 0]
        last_reward = batch.last_reward[:, 0]

        phi = self.phi(params, x, last_action_encoded, last_reward)
        phi_p = self.phi(target, xp, action_encoded, r)

        if self.rep_params.get("frozen"):
            phi = jax.lax.stop_gradient(phi)

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        batch_loss = jax.vmap(partial(q_loss, loss=self.loss), in_axes=0)
        losses, metrics = batch_loss(qs, a, r, gamma, qsp)

        # chex.assert_equal_shape((weights, losses))
        # loss = jnp.mean(weights * losses)
        loss = jnp.mean(losses)

        return loss, metrics

    def policy(self, obs: np.ndarray) -> np.ndarray:
        q = self.values(obs, self.last_action_encoded, self.last_reward)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

    # --------------------------
    # -- Base agent interface --
    # --------------------------
    def values(self, x: np.ndarray, a, r, *args, **kwargs):
        x = np.asarray(x)

        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = np.expand_dims(x, 0)
            q = self._values(self.state, x, a, r, *args, **kwargs)[0]

        else:
            q = self._values(self.state, x, a, r, *args, **kwargs)

        return jax.device_get(q)

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        coeff = self.normalizer_params.get("state", {}).get("coeff")
        if coeff is not None:
            x = 2 * x / coeff - 1
        return x
    
    def encode_action(self, a):
        return jax.nn.one_hot(a, self.num_actions)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, x: np.ndarray): # type: ignore
        self.is_successful = False

        self.last_action_encoded = self.encode_action(jnp.int32(-1))
        self.last_reward = 0.0
        self.buffer.flush()
        x = np.asarray(x)
        x = self.normalize_state(x)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.buffer.add_step(Timestep(
            x=x,
            a=a,
            r=None,
            gamma=self.gamma,
            terminal=False,
            extra={
                'carry': False,
                'carryp': False,
                'reset': True,
                'pos': (-1, -1),
                'last_action_encoded': self.last_action_encoded,
                'last_reward': self.last_reward
                }
        ))
        
        self.last_action_encoded = self.encode_action(jnp.int32(a))

        return a

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]): # type: ignore
        a = -1

        self.last_reward = r

        # sample next action
        if xp is not None:
            xp = np.asarray(xp)
            xp = self.normalize_state(xp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get('gamma', 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)
            
        pos = extra.get('pos', None)

        self.buffer.add_step(Timestep(
            x=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
            terminal=False,
            extra={
                'carry': False,
                'carryp': False,
                'reset': False,
                'pos': pos,
                'last_action_encoded': self.last_action_encoded,
                'last_reward': self.last_reward
                }
        ))

        self.last_action_encoded = self.encode_action(jnp.int32(a))

        self.update()
        return a

    def end(self, r: float, extra: Dict[str, Any]): # type: ignore
        self.is_successful = extra.get('success', False)
        self.last_reward = r
        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        pos = extra.get('pos', None)
        
        self.buffer.add_step(Timestep(
            x=np.zeros(self.observations),
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
            extra={
                'carry': False,
                'carryp': False,
                'reset': False,
                'pos': pos,
                'last_action_encoded': self.last_action_encoded,
                'last_reward': self.last_reward
                }
        ))

        self.update()
