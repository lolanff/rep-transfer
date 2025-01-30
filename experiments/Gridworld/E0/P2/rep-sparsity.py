import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.functions import compute_feature_sparsity
import pickle
import lzma
from experiment.tools import parseCmdLineArgs
import jax.numpy as jnp
import numpy as np


setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()
agent_label = "DQN-ReLU-A"
COLORS = ['r','y','g','b','m']

#---- Input Params ----#
alpha = 0.0001
IDX = np.arange(1, 10, 2)
#----------------------#

def main(): 
    # Load state samples from saved transition data
    state_sample = np.load("analysis/Gridworld/data/transition_current_states.npy")

    # Load experiment params
    exp = ExperimentModel.load(f'{path}/{agent_label}.json')

    seed = 0
    f, ax = plt.subplots()
    for i in IDX:
        # Load checkpoint
        chk = Checkpoint(exp, i)
        with lzma.open(f'checkpoints/results/Gridworld/E0/P1/{agent_label}/{i}/chk.pkl.xz', 'rb') as f:
            chk._storage = pickle.load(f)
        Agent = chk["a"]
        
        sparsity = []
        for state in state_sample: 
            state = jnp.expand_dims(state, axis=0) 
            features = Agent.phi(Agent.state.params, state).out 
            sparsity.append(compute_feature_sparsity(features=features))
      
        ax.hist(sparsity, bins=10, range=(0.5, 1), histtype='stepfilled', 
                color=COLORS[seed], alpha=0.3, label=f'seed={seed}, mean={round(np.mean(sparsity),2)}')
        
        seed += 1
        
    ax.set_title(f'Distribution of Rep Sparsity. {agent_label}. alpha={alpha}')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Sparsity')
    ax.legend()

    save(save_path=f'{path}/plots', plot_name=f'{agent_label}-{alpha}alpha')
    plt.show()

if __name__ == '__main__':
    main()