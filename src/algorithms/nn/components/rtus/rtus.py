# From esraaelelimy/continuing_ppo
from flax import linen as nn
import jax 
import jax.numpy as jnp 
import flax 
from typing import Callable, Any, Tuple, Iterable,Optional
from algorithms.nn.components.rtus.rtus_utils import *
from algorithms.nn.components.rtus.linear_rtus import *


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  
Array = Any

'''
A Consice interface to Real-Time Linear RTUs
Linear recurrence + non-linear output 
'''
class RTLRTUs(nn.Module):
    n_hidden: int   # number of hidden features
    params_type: str = 'exp_exp' # direct, exp, exp_exp_nu, exp_exp
    stable_r: bool = False      # if True, clip r to be \in (eps,1]
    d_input: int = 1
    activation: str = 'relu'
    @nn.compact
    def __call__(self,carry,x_t):
        update_gate = RealTimeLinearRTUs(self.n_hidden,self.params_type,self.stable_r)
        carry,(h_t_c1,h_t_c2)  = update_gate(carry,x_t)
        h_t = act_options[self.activation](jnp.concatenate((h_t_c1, h_t_c2), axis=-1))
        return carry,h_t # carry, output
    def initialize_state(self,batch_size=1):
        hidden_init = (jnp.zeros((batch_size,self.n_hidden)),jnp.zeros((batch_size,self.n_hidden)))
        memory_grad_init = (jnp.zeros((batch_size,self.n_hidden)),jnp.zeros((batch_size,self.n_hidden)),
                            jnp.zeros((batch_size,self.n_hidden)),jnp.zeros((batch_size,self.n_hidden)),
                            jnp.zeros((batch_size,self.d_input, self.n_hidden)),jnp.zeros((batch_size,self.d_input, self.n_hidden)),
                            jnp.zeros((batch_size,self.d_input, self.n_hidden)),jnp.zeros((batch_size,self.d_input, self.n_hidden)))
        return (hidden_init,memory_grad_init)
