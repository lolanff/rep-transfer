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

#------ Input Params ------#
agent_label = "DQN-ReLU-A"
alpha = 0.0001
IDX = np.arange(1, 10, 2)
#--------------------------#

def main(): 
    start = time.time()

    # Load state samples and remove duplicates
    state_sample = np.load("analysis/Gridworld/data/transition_current_states.npy")
    unique_flat_samples = np.unique(state_sample.reshape(1044, -1), axis=0)
    state_sample = unique_flat_samples.reshape(-1, 15, 15, 3)

    # Load experiment params
    exp = ExperimentModel.load(f'experiments/Gridworld/E0/P1/{agent_label}.json')

    seed = 0
    f, ax = plt.subplots()
    COLORS = ['r','y','g','b','m']
    for i in IDX:
        print(f'Start seed {seed}')

        # Load checkpoint
        chk = Checkpoint(exp, i)
        with lzma.open(f'checkpoints/results/Gridworld/E0/P1/{agent_label}/{i}/chk.pkl.xz', 'rb') as f:
            chk._storage = pickle.load(f)
        Agent = chk["a"]

        # Compute features and action values
        phi = Agent.phi(Agent.state.target_params, state_sample).out 
        q = Agent.q(Agent.state.target_params, phi)
        v = jnp.max(q, axis=1)

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
        #Ds = Ds / np.max(Ds)
        #Dv = Dv / np.max(Dv)

        # Plot histogram
        #ratio = Dv/(Ds + 1e-10)
        ratio = Dv / Ds
        ratio = ratio[~np.isnan(ratio)] # remove nan
        hist_values, bin_edges = np.histogram(ratio, bins=20, density=True)
        ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist_values, 
                 color=COLORS[seed], alpha=0.6, label=f'seed={seed}, mean={np.mean(ratio):.4f}')
        seed += 1
        
    ax.set_title(f'Distribution of Rep Lipschitz Ratio. {agent_label}. alpha={alpha}')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Lipschitz Ratio (Dv/Ds)')
    ax.set_xlim([0, 0.4]) # type: ignore
    ax.legend()

    save(save_path=f'{path}/plots', plot_name=f'{agent_label}-{alpha}alpha')
    plt.show()

    print(f"Execution time: {round((time.time()-start),0)} seconds.")

if __name__ == '__main__':
    main()