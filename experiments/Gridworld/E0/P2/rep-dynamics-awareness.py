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
import jax
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

    # Load state samples from saved transition data
    current_states = np.load("results/Gridworld/transition-data/distance_current_states.npy")
    next_states = np.load("results/Gridworld/transition-data/distance_next_states.npy")

    # Create a random state samples by permutation
    np.random.seed(0)
    permuted_indices = np.random.permutation(current_states.shape[0])
    random_states = current_states[permuted_indices]

    # Remove cases where current=next (i.e. walking into wall in gridworld)
    # or current=random (i.e. the random state was a bad choice)
    is_zero = np.all((next_states-current_states == 0) | (random_states-current_states == 0), axis=(1, 2, 3))
    print(f'There are {np.sum(is_zero)} samples where current=next or current=random.')
    current_states = current_states[~is_zero]
    next_states = next_states[~is_zero]
    random_states = random_states[~is_zero]

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

        # Compute features
        current_features = Agent.phi(Agent.state.params, current_states).out 
        next_features = Agent.phi(Agent.state.params, next_states).out 
        random_features = Agent.phi(Agent.state.params, random_states).out 

        # Compute feature distances
        dist_successor = jnp.linalg.norm(next_features - current_features, axis=1)
        dist_random = jnp.linalg.norm(random_features - current_features, axis=1)

        # Compute Han et al's dynamics awareness
        DA = 1 - (jnp.sum(dist_successor)/jnp.sum(dist_random)).item()

        # Plot histogram of dist_successor/dist_random
        da = dist_successor/dist_random
        ax.hist(da, bins=40, range=(0,10), histtype='stepfilled', 
                color=COLORS[seed], alpha=0.3, label=f'seed={seed}, mean={round(jnp.mean(da).item(),3)}, dynamics awareness={round(DA,3)}')
    
        seed += 1
        
    ax.set_title(f'Distribution of dist_successor/dist_random. {agent_label}. alpha={alpha}')
    ax.set_ylabel('Counts')
    ax.set_xlabel('dist_successor/dist_random')
    ax.legend()

    save(save_path=f'{path}/plots', plot_name=f'{agent_label}-{alpha}alpha')
    plt.show()

    print(f"Execution time: {time.time()-start} seconds.")

if __name__ == '__main__':
    main()