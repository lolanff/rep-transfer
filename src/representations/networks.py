import math 
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from algorithms.nn.components.rtus.rtus import RTLRTUs
import haiku.experimental.flax as hkflax
import utils.hk as hku
from utils.functions import fta

ModuleBuilder = Callable[[], Callable[[jax.Array | np.ndarray], jax.Array]]

class GRU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        self.gru = hk.GRU(self.hidden, name='gru_inner', w_h_init=xavier, w_i_init=xavier)
        self.learn_initial_h = learn_initial_h
        
    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter("initial_h", shape=(self.hidden,), init=jnp.zeros)
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.repeat(self.gru.initial_state(batch_size=batch)[:, None, :], length, axis=1)
        return init_h
        
    def gru_step(self, prev_state, inputs):
        frame_feat, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru(frame_feat[None, :], prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(self.gru_step, carry_seq[:1, :], (features_seq, reset_seq, carry_seq))
        return outputs_seq, state_seq
    
    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        
        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]
        
        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...] and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(x, reset, carry)

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]

class TMazeGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        h = self.flatten(x)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class MazeARNet(hk.Module):
    """
    Similar to MazeNetReLU but also accepts last action `a` and last reward `r`.
    These are concatenated to the right of the flattened conv features before the final linear layer.
    """
    def __init__(self, hidden: int, actions: int, use_fta_with_eta=None, name: str = "MazeARNet"):
        super().__init__(name=name)
        self.hidden = hidden
        self.actions = actions
        if use_fta_with_eta is not None:
            self.final_activation = lambda x: fta(x, eta=use_fta_with_eta, tiles=20, lower_bound=-2, upper_bound=2)
        else:
            self.final_activation = jax.nn.relu
        xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv1')
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv2')
        self.flatten = hk.Flatten(name='flatten')
        self.linear = hk.Linear(hidden, w_init=xavier, name='linear')
        self.phi = hk.Flatten(name='phi')

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, r: jnp.ndarray = None, *args, **kwargs):
        # x expected shape: [N, H, W, C]
        N = x.shape[0]

        # Default zero action / reward when not provided (useful for init-time calls)
        if a is None:
            a = jnp.zeros((N, self.actions), dtype=jnp.float32)
        if (len(a.shape) < 2):
            a = jnp.broadcast_to(a, (N, self.actions))
        if r is None:
            r = jnp.zeros((N, 1), dtype=jnp.float32)
        if (len(r.shape) < 2):
            r = r[..., None]
            r = jnp.broadcast_to(r, (N, 1))

        h = self.conv1(x)
        h = jax.nn.relu(h)
        h = self.conv2(h)
        h = jax.nn.relu(h)

        h = self.flatten(h)  # shape [N, feat]

        # Concatenate action and reward to the right of conv features
        h = jnp.concatenate((h, a, r), axis=-1)

        h = self.linear(h)

        h = self.final_activation(h)
        
        out = self.phi(h)

        return out

class MazeGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv1')

        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv2')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')

        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
            
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv1(x)
        h = jax.nn.relu(h)
        h = self.conv2(h)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
 
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)

        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry
    
class MazeARGRUNet(hk.Module):
    def __init__(self, hidden: int, actions: int, use_fta_with_eta=None, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.number_of_actions = actions
        if use_fta_with_eta is not None:
            self.final_activation = lambda x: fta(x, eta=use_fta_with_eta, tiles=20, lower_bound=-2, upper_bound=2)
        else:
            self.final_activation = jax.nn.relu

        xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv1')

        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv2')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')
        
        self.recast = hk.Linear(self.hidden, w_init=xavier, name='recast')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=xavier, name='skip_connection')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, r: jnp.ndarray = None, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        # Populate a if None
        if a is None:
            a = jnp.full((N, T, self.number_of_actions), jnp.float32(0))
        if (len(a.shape) < 3):
            a = jnp.broadcast_to(a, (N, T, self.number_of_actions))
        if r is None:
            r = jnp.full((N, T, 1), jnp.float32(0))
        if (len(r.shape) < 3):
            r = r[..., None]
            r = jnp.broadcast_to(r, (N, T, 1))
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv1(x)
        h = jax.nn.relu(h)
        
        h = self.conv2(h)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        h = jnp.concatenate((h, a, r), axis=-1)
        
        h = self.recast(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = outputs_sequence + self.skip_connection(h)
        
        outputs_sequence = self.final_activation(outputs_sequence)

        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class MazeARRTUNet(hk.Module):
    def __init__(self, hidden: int, actions: int, use_fta_with_eta=None, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.number_of_actions = actions
        if use_fta_with_eta is not None:
            self.final_activation = lambda x: fta(x, eta=use_fta_with_eta, tiles=20, lower_bound=-2, upper_bound=2)
        else:
            self.final_activation = jax.nn.relu

        xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv1')

        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv2')

        self.flatten = hk.Flatten(name='flatten')
        
        self.recast = hk.Linear(self.hidden, w_init=xavier, name='recast')
        
        self.skip_connection = hk.Linear(self.hidden, w_init=xavier, name='skip_connection')

        self.rtu = hkflax.lift(RTLRTUs(int(self.hidden/2), d_input=self.hidden, params_type='exp_exp', name='rtu'), name='lifted_rtu')
        
        self.phi = hk.Flatten(name='phi')
    
    def initialize_state(self,batch_size=1):
        n_hidden = int(self.hidden/2) # h_t_c1, h_t_c2 are concatenated to form h_t of size hidden
        d_input = self.hidden
        hidden_init = (jnp.zeros((batch_size,n_hidden)),jnp.zeros((batch_size,n_hidden)))
        memory_grad_init = (jnp.zeros((batch_size,n_hidden)),jnp.zeros((batch_size,n_hidden)),
                            jnp.zeros((batch_size,n_hidden)),jnp.zeros((batch_size,n_hidden)),
                            jnp.zeros((batch_size,d_input, n_hidden)),jnp.zeros((batch_size,d_input, n_hidden)),
                            jnp.zeros((batch_size,d_input, n_hidden)),jnp.zeros((batch_size,d_input, n_hidden)))
        return (hidden_init,memory_grad_init)

    def __call__(self, x: jnp.ndarray, a: jnp.ndarray = None, r: jnp.ndarray = None, carry: jnp.ndarray = None, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray, Any]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        
        N, *feat = x.shape
        
        # Populate a if None
        if a is None:
            a = jnp.full((N, self.number_of_actions), jnp.float32(0))
        if (len(a.shape) < 2):
            a = jnp.broadcast_to(a, (N, self.number_of_actions))
        if r is None:
            r = jnp.full((N, 1), jnp.float32(0))
        if (len(r.shape) < 2):
            r = r[..., None]
            r = jnp.broadcast_to(r, (N, 1))
        if carry is None:
            carry = self.initialize_state(N)

        h = self.conv1(x)
        h = jax.nn.relu(h)
        
        h = self.conv2(h)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = self.flatten(h)
        
        h = jnp.concatenate((h, a, r), axis=-1)
        
        h = self.recast(h)

        carry, output = self.rtu(carry, h)
        
        output = output #+ self.skip_connection(h)
        
        output = self.final_activation(output)

        output = self.phi(output)

        return output, carry, self.initialize_state()

class MazeGRUNetFTA(hk.Module):
    def __init__(self, hidden: int, eta, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.eta = eta
        xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")

        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv1')

        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv2')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
            
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv1(x)
        h = jax.nn.relu(h)
        h = self.conv2(h)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        N, T, _ = outputs_sequence.shape
        
        outputs_sequence = jnp.reshape(outputs_sequence, (N * T, -1))
        
        outputs_sequence = fta(outputs_sequence, eta=self.eta, tiles=20, lower_bound=-2, upper_bound=2)
        
        outputs_sequence = jnp.reshape(outputs_sequence, (N, T, -1))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 1, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerGRUNetFTA(hk.Module):
    def __init__(self, hidden: int, eta, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.eta = eta
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 1, w_init=w_init, name='phi')

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, learn_initial_h=learn_initial_h, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None, is_target = False) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
        
        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if (len(x.shape) < 5):
            x = x[:, None]
        
        N, T, *feat = x.shape
        
        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)
        
        _, *feat = h.shape
        
        h = jnp.reshape(h, (N, T, *feat))
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry, is_target=is_target)
        
        N, T, _ = outputs_sequence.shape
        
        outputs_sequence = jnp.reshape(outputs_sequence, (N * T, -1))
        
        outputs_sequence = fta(outputs_sequence, eta=self.eta, tiles=20, lower_bound=-2, upper_bound=2)
        
        outputs_sequence = jnp.reshape(outputs_sequence, (N, T, -1))
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class NetworkBuilder:
    def __init__(self, input_shape: Tuple, params: Dict[str, Any], seed: int):
        self._input_shape = tuple(input_shape)
        self._h_params = params
        self._rng, feat_rng = jax.random.split(jax.random.PRNGKey(seed))

        self._feat_net, self._feat_params = buildFeatureNetwork(input_shape, params, feat_rng)

        self._params = {
            'phi': self._feat_params,
        }

        self._retrieved_params = False

    def getParams(self):
        self._retrieved_params = True
        return self._params

    def getFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray):
            return self._feat_net.apply(params['phi'], x)

        return _inner
    
    def getActionRewardFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, a: jax.Array, r: jax.Array):
            return self._feat_net.apply(params['phi'], x, a, r)

        return _inner

    def getRecurrentFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, reset: jax.Array | np.ndarray = None, carry: jax.Array | np.ndarray = None, is_target = False):
            return self._feat_net.apply(params['phi'], x, reset=reset, carry=carry, is_target=is_target)

        return _inner
    
    def getActionRewardRecurrentFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, a: jax.Array, r: jax.Array, reset: jax.Array | np.ndarray = None, carry: jax.Array | np.ndarray = None, is_target = False):
            return self._feat_net.apply(params['phi'], x, a, r, reset=reset, carry=carry, is_target=is_target)

        return _inner

    def addHead(self, module: ModuleBuilder, name: Optional[str] = None, grad: bool = True):
        assert not self._retrieved_params, 'Attempted to add head after params have been retrieved'
        _state = {}

        def _builder(x: jax.Array | np.ndarray):
            head = module()
            _state['name'] = getattr(head, 'name', None)

            if not grad:
                x = jax.lax.stop_gradient(x)

            out = head(x)
            return out

        sample_in = jnp.zeros((1,) + self._input_shape)
        if 'GRU' in self._h_params['type']:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in)[0]
        if 'RTU' in self._h_params['type']:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in)[0]
        elif 'AR' in self._h_params['type']:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in)
        else:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in).out

        self._rng, rng = jax.random.split(self._rng)
        h_net = hk.without_apply_rng(hk.transform(_builder))
        h_params = h_net.init(rng, sample_phi)

        name = name or _state.get('name')
        assert name is not None, 'Could not detect name from module'
        self._params[name] = h_params

        def _inner(params: Any, x: jax.Array):
            return h_net.apply(params[name], x)

        return _inner


def reluLayers(layers: List[int], name: Optional[str] = None):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=w_init, b_init=b_init, name=name))
        out.append(jax.nn.relu)

    return out

def buildFeatureNetwork(inputs: Tuple, params: Dict[str, Any], rng: Any):
    def _inner(x: jax.Array, *args, **kwargs):
        name = params['type']
        hidden = params['hidden']

        if name == 'TwoLayerRelu':
            layers = reluLayers([hidden, hidden], name='phi')

        elif name == 'OneLayerRelu':
            layers = reluLayers([hidden], name='phi')

        elif name == 'MinatarNet':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 2, w_init=w_init, name='phi'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            layers += reluLayers([hidden], name='phi')

        elif name == 'ForagerNetReLU':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name='phi'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            layers += reluLayers([hidden], name='phi')

        elif name == 'ForagerGRUNetReLU':
            # It uses initializer different from above
            net = ForagerGRUNetReLU(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetReLU')
            return net(x, *args, **kwargs)
        
        elif name == 'ForagerGRUNetFTA':
            net = ForagerGRUNetFTA(hidden=hidden, eta=params['eta'], learn_initial_h=params.get('learn_initial_h', True), name='ForagerGRUNetFTA')
            return net(x, *args, **kwargs)

        elif name == 'AtariNet':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                lambda x: x.astype(np.float32),
                make_conv(32, (8, 8), (4, 4)),
                jax.nn.relu,
                make_conv(64, (4, 4), (2, 2)),
                jax.nn.relu,
                make_conv(64, (3, 3), (1, 1)),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(512, w_init=w_init),
                jax.nn.relu,
            ]
        
        elif name == 'MazeNetReLU':
            # Use Pytorch default initialization for Conv2d
            # see https://github.com/pytorch/pytorch/blob/9bc9d4cdb4355a385a7d7959f07d04d1648d6904/torch/nn/modules/conv.py#L178
            xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
            layers = [
                hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv'),
                jax.nn.relu,
                hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv_1'),
                jax.nn.relu,
                hk.Flatten(name='flatten'),
                hk.Linear(hidden, w_init=xavier, name='linear'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            
        elif name == 'MazeARNet':
            net = MazeARNet(hidden=hidden, actions=params['actions'], use_fta_with_eta=params.get('use_fta_with_eta', None), name='MazeARNet')
            return net(x, *args, **kwargs)

        elif name == 'MazeNetFTA':
            # https://github.com/pytorch/pytorch/blob/9bc9d4cdb4355a385a7d7959f07d04d1648d6904/torch/nn/modules/conv.py#L178
            xavier = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
            layers = [
                hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=xavier, name='conv'),
                jax.nn.relu,
                hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=xavier, name='conv_1'),
                jax.nn.relu,
                hk.Flatten(name='flatten'),
                hk.Linear(hidden, w_init=xavier, name='linear'),
                lambda x: fta(x, eta=params['eta'], tiles=20, lower_bound=-2, upper_bound=2),
                hk.Flatten(name='phi'),
            ]
           
        elif name == 'TMazeGRUNetReLU':
            net = TMazeGRUNetReLU(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='TMazeGRUNetReLU')
            return net(x, *args, **kwargs)
         
        elif name == 'MazeGRUNetReLU':
            net = MazeGRUNetReLU(hidden=hidden, learn_initial_h=params.get('learn_initial_h', True), name='MazeGRUNetReLU')
            return net(x, *args, **kwargs)
        
        elif name == 'MazeARGRUNet':
            net = MazeARGRUNet(hidden=hidden, actions=params['actions'], use_fta_with_eta=params.get('use_fta_with_eta', None), learn_initial_h=params.get('learn_initial_h', True), name='MazeARGRUNet')
            return net(x, *args, **kwargs)
        
        elif name == 'MazeGRUNetFTA':
            net = MazeGRUNetFTA(hidden=hidden, eta=params['eta'], learn_initial_h=params.get('learn_initial_h', True), name='MazeGRUNetFTA')
            return net(x, *args, **kwargs)

        elif name == 'MazeARRTUNet':
            net = MazeARRTUNet(hidden=hidden, actions=params['actions'], use_fta_with_eta=params.get('use_fta_with_eta', None), learn_initial_h=params.get('learn_initial_h', True), name='MazeARRTUNet')
            return net(x, *args, **kwargs)
        
        elif name == 'Linear':
            layers = [
                hk.Flatten(name='flatten'),
                hk.Linear(hidden, name='linear'),
                hk.Flatten(name='phi'),
            ]

        else:
            raise NotImplementedError()

        return hku.accumulatingSequence(layers)(x)

    network = hk.without_apply_rng(hk.transform(_inner))

    sample_input = jnp.zeros((1,) + tuple(inputs))
    net_params = network.init(rng, sample_input)

    return network, net_params


def make_conv(size: int, shape: Tuple[int, int], stride: Tuple[int, int]):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)
    return hk.Conv2D(
        size,
        kernel_shape=shape,
        stride=stride,
        w_init=w_init,
        b_init=b_init,
        padding='VALID',
        name='conv',
    )
