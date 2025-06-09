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
import pandas as pd
import jax

setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

# Input params
chk_path = ['checkpoints/results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQN-FTA-Gridworld',
            'checkpoints/results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQN-ReLU-Gridworld',
            'checkpoints/results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQNAux-FTA-Gridworld',
            'checkpoints/results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQNAux-ReLU-Gridworld']

idx = [0, 1, 2, 3, 4]    # enter the checkpoint indices

batch_path = 'experiments/Gridworld/A3/P5/sampled_buffer.pkl.xz'

with lzma.open(batch_path, 'rb') as f:
    batch = pickle.load(f)

def main():
    df = pd.DataFrame(columns=['algorithm', 'seed', 'Raw Complexity Reduction', 'Dynamics Awareness', 'Diversity', 'Orthogonality', 'Sparsity'])

    x = batch.x
    xp = batch.xp
    reset = batch.reset
    
    def add_entry(df, algorithm, seed, cr, da, d, o, s):
        new_entry = pd.DataFrame([[algorithm, seed, cr, da, d, o, s]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)
    
    for alg in chk_path:
        for i in idx:
            # Load the agent in the checkpoint
            with lzma.open(alg + f'/{i}/chk.pkl.xz', 'rb') as f:
                chk_storage = pickle.load(f)
            agent = chk_storage["a"]

            phi = agent.phi(agent.state.params, x, reset=reset)[0][:,-1]
            phi_p = agent.phi(agent.state.params, xp, reset=np.concatenate((reset[:, 1:], np.zeros((reset.shape[0],1), dtype=bool)), axis=1))[0][:,-1]
            
            q = agent.q(agent.state.params, phi)
            v = jnp.max(q, axis=1)

            cr = complexity_reduction_unnormalized(phi, v)
            print(f'Complexity reduction of Rep #{i} is {cr}.')
            da = dynamics_awareness(phi, phi_p)
            print(f'Dynamics awareness of Rep #{i} is {da}.')
            d = diversity(phi, v)
            print(f'Diversity of Rep #{i} is {d}.')
            o = orthogonality(phi)
            print(f'Orthogonality of Rep #{i} is {o}.')
            s = sparsity(phi)
            print(f'Sparsity of Rep #{i} is {s}.')
            
            df = add_entry(df, alg.split('/')[-1], i, cr, da, d, o, s)
        
    # Renormalize CR
    df['Complexity Reduction'] = 1 - df['Raw Complexity Reduction'] / df['Raw Complexity Reduction'].max()

    df.to_csv(f"{path}/results.csv")
 
def complexity_reduction_unnormalized(phi, v):   
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

    return np.mean(ratio)

def dynamics_awareness(phi, phi_p):   
    # Create a random phi samples by permuting phi
    np.random.seed(0)
    permuted_indices = np.random.permutation(phi.shape[0])
    phi_r = phi[permuted_indices]

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