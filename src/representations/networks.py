import math 
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

import utils.hk as hku
from utils.functions import fta

ModuleBuilder = Callable[[], Callable[[jax.Array | np.ndarray], jax.Array]]

class GRU(hk.Module):
    def __init__(self, hidden: int, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.gru = hk.GRU(self.hidden, name='gru_inner')
        
    def gru_step(self, prev_state, inputs):
        frame_feat, reset_flag = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, self.gru.initial_state(batch_size=1), prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru(frame_feat[None, :], prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, carry, features_seq, reset_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(self.gru_step, carry[None, :], (features_seq, reset_seq))
        return outputs_seq, state_seq
    
    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        
        N, T = x.shape[0], x.shape[1]
        
        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)
        if carry is None:
            carry = self.gru.initial_state(batch_size=N)
            
        # Shift reset
        reset = jnp.hstack((jnp.zeros((N, 1), dtype=bool), reset[:, :-1]))

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...] and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(carry, x, reset)

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.gru.initial_state(batch_size=1)

class TMazeGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden

        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class MazeGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        w_conv_init = hk.initializers.VarianceScaling(math.sqrt(5), "fan_avg", "uniform")
        b_conv_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")

        self.conv1 = hk.Conv3D(
            output_channels=32,
            kernel_shape=(1, 4, 4),
            stride=1,
            padding=[(0, 0), (1, 1), (1, 1)],
            w_init=w_conv_init,
            b_init=b_conv_init,
            name="conv_1"
        )

        self.conv2 = hk.Conv3D(
            output_channels=16,
            kernel_shape=(1, 4, 4),
            stride=(1, 2, 2),
            padding=[(0, 0), (2, 2), (2, 2)],
            w_init=w_conv_init,
            b_init=b_conv_init,
            name="conv_2"
        )
        self.flatten = hk.Flatten(preserve_dims=2, name='flatten')

        self.gru = GRU(self.hidden, name='gru')
        
        self.phi = hk.Flatten(preserve_dims=2, name='phi')

    def __call__(self, x: jnp.ndarray, reset: jnp.ndarray = None, carry: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray]:
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

        h = self.conv1(x)
        h = jax.nn.relu(h)
        h = self.conv2(h)
        h = jax.nn.relu(h)
        
        h = self.flatten(h)
        
        outputs_sequence, states_sequence, initial_carry = self.gru(h, reset, carry)
        
        outputs_sequence = jax.nn.relu(outputs_sequence)
        
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

    def getRecurrentFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, reset: jax.Array | np.ndarray = None, carry: jax.Array | np.ndarray = None):
            return self._feat_net.apply(params['phi'], x, reset=reset, carry=carry)

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

        elif name == 'ForagerNet':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 2, w_init=w_init, name='phi'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            layers += reluLayers([hidden], name='phi')

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
            w_conv_init = hk.initializers.VarianceScaling(math.sqrt(5), "fan_avg", "uniform")
            b_conv_1_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
            layers = [
                hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=w_conv_init, b_init=b_conv_1_init, name='conv'),
                jax.nn.relu,
                hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=w_conv_init, b_init=b_conv_1_init, name='conv_1'),
                jax.nn.relu,
                hk.Flatten(name='flatten'),
                hk.Linear(hidden, w_init=w_init, name='linear'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]

        elif name == 'MazeNetFTA':
            # https://github.com/pytorch/pytorch/blob/9bc9d4cdb4355a385a7d7959f07d04d1648d6904/torch/nn/modules/conv.py#L178
            w_conv_init = hk.initializers.VarianceScaling(math.sqrt(5), "fan_avg", "uniform")
            b_conv_1_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
            layers = [
                hk.Conv2D(output_channels=32, kernel_shape=4, stride=1, padding=[(1, 1)], w_init=w_conv_init, b_init=b_conv_1_init, name='conv'),
                jax.nn.relu,
                hk.Conv2D(output_channels=16, kernel_shape=4, stride=2, padding=[(2, 2)], w_init=w_conv_init, b_init=b_conv_1_init, name='conv_1'),
                jax.nn.relu,
                hk.Flatten(name='flatten'),
                hk.Linear(hidden, name='linear'),  # What's a suitable weight/bias initializer for FTA?
                lambda x: fta(x, eta=params['eta'], tiles=20, lower_bound=-2, upper_bound=2),
                hk.Flatten(name='phi'),
            ]
           
        elif name == 'TMazeGRUNetReLU':
            net = TMazeGRUNetReLU(hidden=hidden, name='TMazeGRUNetReLU')
            return net(x, *args, **kwargs)
         
        elif name == 'MazeGRUNetReLU':
            net = MazeGRUNetReLU(hidden=hidden, name='MazeGRUNetReLU')
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
