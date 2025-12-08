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
chk_path = [
                'checkpoints/results/Gridworld/A4/P2/gridworldpartial_dqn_pretrain/DQN-FTA-GridworldPartial',
                'checkpoints/results/Gridworld/A4/P2/gridworldpartial_dqn_pretrain/DQN-ReLU-GridworldPartial',
                'checkpoints/results/Gridworld/A4/P2/gridworldpartial_drqn_2_16_pretrain/DRQN-FTA-GridworldPartial',
                'checkpoints/results/Gridworld/A4/P2/gridworldpartial_drqn_2_16_pretrain/DRQN-ReLU-GridworldPartial'
            ]

idx = [0, 1, 2, 3, 4]    # enter the checkpoint indices

batch_path = 'experiments/Gridworld/A4/P7/sampled_buffer_alt.pkl.xz'

with lzma.open(batch_path, 'rb') as f:
    batch = pickle.load(f)

def main():
    df = pd.DataFrame(columns=['algorithm', 'seed', 'State Awareness'])

    traj = []
    for seq_pair in batch:
        traj.append(seq_pair.x)
    traj = jnp.array(traj)
    print(traj.shape)
    
    def add_entry(df, algorithm, seed, sa):
        new_entry = pd.DataFrame([[algorithm, seed, sa]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)
    
    for alg in chk_path:
        for i in idx:
            # Load the agent in the checkpoint
            with lzma.open(alg + f'/{i}/chk.pkl.xz', 'rb') as f:
                chk_storage = pickle.load(f)
            agent = chk_storage["a"]

            if 'drqn' in alg:
                phi_t1 = agent.phi(agent.state.params, traj[:, 0])[0][:,-1]
                phi_t2 = agent.phi(agent.state.params, traj[:, 1])[0][:,-1]
                phi_t3 = agent.phi(agent.state.params, traj[:, 2])[0][:,-1]
            else:
                phi_t1 = agent.phi(agent.state.params, traj[:, 0, -1]).out 
                phi_t2 = agent.phi(agent.state.params, traj[:, 1, -1]).out 
                phi_t3 = agent.phi(agent.state.params, traj[:, 2, -1]).out


            sa = state_awareness(phi_t1, phi_t2, phi_t3)
            print(f'State awareness of Rep #{i} is {sa}.')
            
            df = add_entry(df, alg.split('/')[-1], i, sa)
        
    df.to_csv(f"{path}/results_alt.csv")
 
def dynamics_awareness(phi, phi_p):   
    # Create a random phi samples by permuting phi
    np.random.seed(0)
    permuted_indices = np.random.permutation(phi.shape[0])
    phi_r = phi[permuted_indices]

    # Compute feature distances
    dist_successor = jnp.linalg.norm(phi_p - phi, axis=1)
    dist_random = jnp.linalg.norm(phi_r - phi, axis=1)

    return 1 - (jnp.sum(dist_successor)/jnp.sum(dist_random)).item()

def state_awareness(phi_t1, phi_t2, phi_t3):
    # Compute feature distances
    dist_same = jnp.linalg.norm(phi_t1 - phi_t2, axis=1)
    dist_other = jnp.linalg.norm(phi_t1 - phi_t3, axis=1)

    return (np.sum(dist_other) - np.sum(dist_same)) / (np.sum(dist_other) + 1e-10).item()

if __name__ == '__main__':
    main()