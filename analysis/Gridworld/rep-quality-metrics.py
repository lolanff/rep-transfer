import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
import pickle
import lzma
from experiment.tools import parseCmdLineArgs
import jax.numpy as jnp
import numpy as np
import time

setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

# Input params
chk_path = 'checkpoints/results/Gridworld/E0/P1/DQN-ReLU-A'
idx = [0]    # enter the checkpoint indices
Lmax = 1     # Please enter the largest average Lipschitz constant across all reps. Ask Han for the correct value.

# Load state samples from Han et al's repository
current_states = np.load("analysis/Gridworld/data/transition_current_states.npy")
next_states = np.load("analysis/Gridworld/data/transition_next_states.npy")

def main(): 
    for i in idx:
        # Load the agent in the checkpoint
        with lzma.open(chk_path + f'/{i}/chk.pkl.xz', 'rb') as f:
            chk_storage = pickle.load(f)
        agent = chk_storage["a"]

         # Compute features and action values
        phi = agent.phi(agent.state.target_params, current_states).out 
        phi_p = agent.phi(agent.state.target_params, next_states).out 
        q = agent.q(agent.state.target_params, phi)
        v = jnp.max(q, axis=1)

        print(f'Complexity reduction of Rep #{i} is {complexity_reduction(phi, v, Lmax)}.')
        print(f'Dynamics awareness of Rep #{i} is {dynamics_awareness(agent, phi, phi_p)}.')
        print(f'Diversity of Rep #{i} is {diversity(phi, v)}.')
        print(f'Orthogonality of Rep #{i} is {orthogonality(phi)}.')
        print(f'Sparsity of Rep #{i} is {sparsity(phi)}.')
 
def complexity_reduction(phi, v, Lmax):   
    # Compute pairwise distances
    Ds = []
    Dv = []
    for i in range(phi.shape[0]):
        for j in range(i):
            Ds.append(jnp.linalg.norm(phi[i] - phi[j]))
            Dv.append(jnp.abs(v[i] - v[j]))
    Ds = np.array(Ds)
    Dv = np.array(Dv)

    # Compute ratio
    epsilon = 1e-10  # small value to prevent division by 0 (not explicitly stated in the paper)
    ratio = Dv/(Ds + epsilon) 

    return 1 - np.mean(ratio) / Lmax

def dynamics_awareness(agent, phi, phi_p):   
    # Create a random state samples by permuting "current_states"
    np.random.seed(0)
    permuted_indices = np.random.permutation(current_states.shape[0])
    random_states = current_states[permuted_indices]

    # Compute the features of random states
    phi_r = agent.phi(agent.state.params, random_states).out 

    # Compute feature distances
    dist_successor = jnp.linalg.norm(phi_p - phi, axis=1)
    dist_random = jnp.linalg.norm(phi_r - phi, axis=1)

    return 1 - (jnp.sum(dist_successor)/jnp.sum(dist_random)).item()

def diversity(phi, v):   
    # Compute pairwise distances
    Ds = []
    Dv = []
    for i in range(phi.shape[0]):
        for j in range(i):
            Ds.append(jnp.linalg.norm(phi[i] - phi[j]))
            Dv.append(jnp.abs(v[i] - v[j]))
    Ds = np.array(Ds)
    Dv = np.array(Dv)

    # Normalize by largest distance
    Ds = Ds / np.max(Ds)
    Dv = Dv / np.max(Dv)
    
    # Compute ratio
    epsilon = 1e-10  # used in Han et al's paper, but the exact value is not given
    ratio = Dv/(Ds + epsilon) 

    return 1 - np.mean(np.minimum(ratio, np.ones(len(ratio))))


def orthogonality(phi):   
    # Normalize the features 
    norms = jnp.linalg.norm(phi, axis=1)
    nonzero_mask = norms > 0  # a mask to filter out vectors with zero norms
    filtered_phi = phi[nonzero_mask]
    filtered_norms = norms[nonzero_mask]
    normalized_phi = filtered_phi / filtered_norms[:, None]

     # Compute pairwise dot products
    cosine_similarities = jnp.dot(normalized_phi, normalized_phi.T)
    upper_triangle_indices = jnp.triu_indices(cosine_similarities.shape[0], k=1)
    upper_triangle_values = cosine_similarities[upper_triangle_indices]

    # Convert to pairwise orthogonalities
    orthogonality = 1 - upper_triangle_values

    return jnp.mean(orthogonality).item()

def sparsity(phi):
    threshold = 1e-10
    total_elements = phi.size
    zero_elements = jnp.count_nonzero(jnp.abs(phi) < threshold).item()
    return zero_elements / total_elements

if __name__ == '__main__':
    main()