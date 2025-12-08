from typing import Any, Dict, Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import utils.chex as cxu
from ml_instrumentation.Collector import Collector
from utils.policies import sample
from utils.hk import MultiLayerHead

from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from ReplayTables.interface import Timestep

@cxu.dataclass
class AgentState:
    params: hk.Params
    optim: optax.OptState

def _ppo_loss(params, batch, phi_apply, policy_apply, value_apply, clip_epsilon, entropy_coef, is_recurrent=False, frozen=False):

    if is_recurrent:
        carry = jax.tree_util.tree_map(lambda x: x.squeeze(1), batch['prev_carry'])
        phi, _, _ = phi_apply(params, batch['obs'], a=batch['prev_a'], r=batch['prev_r'], carry=carry)

    else:
        phi = phi_apply(params, batch['obs'], a=batch['prev_a'], r=batch['prev_r'])
    
    if frozen:
        phi = jax.lax.stop_gradient(phi)

    logits = policy_apply(params, phi)
    logps = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    
    # Ensure actions are correct shape for take_along_axis
    actions = batch['actions']
    if actions.ndim == 1:
        actions = actions[:, None]
        
    sel_logps = jnp.take_along_axis(logps, actions, axis=1).squeeze()
    ratios = jnp.exp(sel_logps - batch['old_logps'])
    clipped = jnp.clip(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = jnp.minimum(ratios * batch['advs'], clipped * batch['advs'])
    policy_loss = -jnp.mean(surrogate)
    
    values = value_apply(params, phi).squeeze()
    
    # Value clipping
    v_clipped = batch['old_values'] + jnp.clip(values - batch['old_values'], -clip_epsilon, clip_epsilon)
    v_loss1 = (batch['returns'] - values) ** 2
    v_loss2 = (batch['returns'] - v_clipped) ** 2
    value_loss = jnp.mean(jnp.maximum(v_loss1, v_loss2))
    
    entropy = -jnp.sum(probs * logps, axis=1).mean()
    
    return policy_loss + 0.5 * value_loss - entropy_coef * entropy

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def _ppo_update_step(state, batch, phi_apply, policy_apply, value_apply, clip_epsilon, entropy_coef, optimizer, is_recurrent, frozen):
    grads = jax.grad(_ppo_loss)(state.params, batch, phi_apply, policy_apply, value_apply, clip_epsilon, entropy_coef, is_recurrent, frozen)
    
    # Check RTU gradients
    for module_name, module_params in grads.items():
        jax.debug.print("Module: {m}", m=module_name)
        print(module_params.items())
        if 'lifted_rtu' in module_name:
            for param_name, param_grad in module_params.items():
                norm = jnp.linalg.norm(param_grad)
                print("RTU Grad {m}/{p}: {n}", m=module_name, p=param_name, n=norm)

    updates, new_opt_state = optimizer.update(grads, state.optim)
    new_params = optax.apply_updates(state.params, updates)

    # Check RTU parameter updates
    for module_name, module_params in state.params.items():
        if 'lifted_rtu' in module_name:
            for param_name, param_val in module_params.items():
                new_val = new_params[module_name][param_name]
                diff = jnp.linalg.norm(new_val - param_val)
                print("RTU Update {m}/{p}: {d}", m=module_name, p=param_name, d=diff)

    return state.replace(params=new_params, optim=new_opt_state)

class PPO(NNAgent):
    """
    Proximal Policy Optimization (PPO) agent stub.
    Implements same interface as other NN agents.
    """
    def __init__(self,
                 observations: Tuple[int, ...],
                 actions: int,
                 params: Dict,
                 collector: Collector,
                 seed: int):
        # Build shared networks via NNAgent (calls PPO._build_heads)
        super().__init__(observations, actions, params, collector, seed)
        self.num_actions = actions
        # PPO-specific hyperparameters
        self.clip_epsilon = params.get('clip_epsilon', 0.2)
        self.gae_lambda = params.get('gae_lambda', 0.95)
        self.ppo_epochs = params.get('ppo_epochs', 4)
        self.rollout_steps = params.get('rollout_steps', 1024)
        self.mini_batch_size = params.get('mini_batch_size', 32)
        self.entropy_coef = params.get('entropy_coef', 0.01)
        self.max_grad_norm = params.get('max_grad_norm', 0.5)

        if self.max_grad_norm is not None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                self.optimizer
            )
            self.state = self.state.replace(optim=self.optimizer.init(self.state.params))

        # on-policy trajectory buffer
        self.trajectory: list = []
        self.episode_ended = False
        
        # Recurrent state
        self.is_recurrent = 'U' in self.rep_params['type'] 
        self.carry = None
        self.init_carry = None
        self.input_action = 0
        self.input_reward = 0.0

    def get_feature_function(self, builder: NetworkBuilder):
        if 'RTU' in self.rep_params['type']:
             return builder.getActionRewardRecurrentFeatureFunction()
        return builder.getActionRewardFeatureFunction()

    def _build_heads(self, builder: NetworkBuilder) -> None:
        # policy head outputs action logits
        self.policy_head = builder.addHead(
            lambda: MultiLayerHead(actions=self.actions, name='policy')
        )
        # value head outputs state-value estimate
        self.value_head = builder.addHead(
            lambda: MultiLayerHead(actions=1, name='value')
        )

    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array) -> jax.Array:  # type: ignore[override]
        """Return state-value estimates for given observations."""
        # Note: This method is not used in the recurrent update path implemented below.
        # It is kept for compatibility or if needed for other purposes.
        phi = self.phi(state.params, x).out
        v = self.value_head(state.params, phi).squeeze(-1)
        return v
    
    def policy(self, obs):
        obs = np.expand_dims(obs, 0)
        
        if self.is_recurrent:
            phi_out = self.phi(self.state.params, obs, a=self.last_action_encoded, r=self.last_reward, carry=self.carry)
            phi = phi_out[0]
            
            next_carry = phi_out[1]
            initial_carry = phi_out[2]
            
            self.carry = next_carry
            self.init_carry = initial_carry

        else:
            phi = self.phi(self.state.params, obs, a=self.last_action_encoded, r=self.last_reward)

        logits = self.policy_head(self.state.params, phi)
        
        value = self.value_head(self.state.params, phi).squeeze()
        probs = jax.nn.softmax(logits).squeeze()
        a = int(sample(np.asarray(probs), rng=self.rng))
        logp = float(jnp.log(probs[a]))
        return a, logp, value
    
    def encode_action(self, a):
        return jax.nn.one_hot(a, self.num_actions)

    def start(self, x: np.ndarray) -> int:  # type: ignore
        self.is_successful = False
        self.episode_ended = False
        self.carry = None

        self.last_action_encoded = self.encode_action(jnp.int32(-1))
        self.last_reward = jnp.asarray(0.0)
        
        # Initialize prev_a and prev_r for trajectory
        self.prev_a = self.last_action_encoded
        self.prev_r = self.last_reward
        
        x = self.normalize_state(np.asarray(x))
        self.last_x = x

        a, logp, value = self.policy(x)

        self.last_carry = self.carry
        self.prev_carry = self.init_carry

        self.last_action_encoded = self.encode_action(a)
        self.last_action, self.last_logp, self.last_value = a, logp, value
        
        return a

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]) -> int:  # type: ignore
        self.episode_ended = False
        # append previous step
        gamma = extra.get('gamma', self.gamma)
        self.last_reward = jnp.asarray(r)
        
        x = self.normalize_state(np.asarray(xp))
        a, logp, value = self.policy(x)

        t = {
            'state': self.last_x,
            'action': self.last_action,
            'logp': self.last_logp,
            'reward': r,
            'gamma': gamma,
            'value': self.last_value,
            'prev_a': self.prev_a,
            'prev_r': self.prev_r,
            'prev_carry': self.prev_carry,
            'next_value': value,
        }
        self.trajectory.append(t)
        
        if len(self.trajectory) >= self.rollout_steps:
            self.update()
            self.trajectory = []

        self.prev_a = self.last_action_encoded
        self.prev_r = self.last_reward
        assert not all(np.allclose(a, b) for a, b in zip(self.last_carry, self.prev_carry))
        self.prev_carry = self.last_carry

        self.last_x = x
        self.last_action_encoded = self.encode_action(a)
        self.last_action, self.last_logp, self.last_value = a, logp, value
        self.last_carry = self.carry
            
        return a

    def end(self, r: float, extra: Dict[str, Any]) -> None:  # type: ignore
        self.episode_ended = True
        gamma = jnp.asarray(0.0)
        self.last_reward = jnp.asarray(r)
        t = {
            'state': self.last_x,
            'action': self.last_action,
            'logp': self.last_logp,
            'reward': r,
            'gamma': gamma,
            'value': self.last_value,
            'prev_a': self.prev_a,
            'prev_r': self.prev_r,
            'prev_carry': self.prev_carry,
            'next_value': jnp.asarray(0.0)
        }
        self.trajectory.append(t)

    def update(self) -> None:
        """Perform PPO epochs on collected trajectory."""
        # gather arrays
        obs = jnp.stack([t['state'] for t in self.trajectory])
        acts = jnp.array([t['action'] for t in self.trajectory])
        r = jnp.array([t['reward'] for t in self.trajectory])
        old_logps = jnp.array([t['logp'] for t in self.trajectory])
        rewards = jnp.array([t['reward'] for t in self.trajectory])
        values = jnp.array([t['value'] for t in self.trajectory])
        gammas = jnp.array([t['gamma'] for t in self.trajectory])
        prev_a = jnp.array([t['prev_a'] for t in self.trajectory])
        prev_r = jnp.array([t['prev_r'] for t in self.trajectory])
        next_values = jnp.array([t['next_value'] for t in self.trajectory])
        
        if self.is_recurrent:
            carries = [t['prev_carry'] for t in self.trajectory]
            carry = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *carries)
        else:
            carry = None

        # compute GAE
        advs = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gammas[i] * next_values[i] - values[i]
            gae = delta + gammas[i] * self.gae_lambda * gae
            advs.insert(0, gae)
        advs = jnp.array(advs)
        returns = advs + values
        
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        batch = {
            'obs': obs,
            'reward': r, 
            'actions': acts, 
            'old_logps': old_logps, 
            'old_values': values,
            'returns': returns, 
            'advs': advs,
            'prev_a': prev_a,
            'prev_r': prev_r,
        }

        if self.is_recurrent:
            batch['prev_carry'] = carry
        

        # update epochs
        num_samples = obs.shape[0]
        indices = np.arange(num_samples)
        
        for _ in range(self.ppo_epochs):
            if not self.is_recurrent:
                self.rng.shuffle(indices)
                
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                idx = indices[start:end]
                mb = jax.tree_util.tree_map(lambda x: x[idx], batch)
                
                self.state = _ppo_update_step(
                    self.state, 
                    mb, 
                    self.phi, 
                    self.policy_head, 
                    self.value_head, 
                    self.clip_epsilon, 
                    self.entropy_coef,
                    self.optimizer,
                    self.is_recurrent,
                    self.rep_params.get("frozen", False)
                )
        # clear trajectory
        self.trajectory = []
