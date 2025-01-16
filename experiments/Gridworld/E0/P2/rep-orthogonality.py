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

    # Load state samples from saved transition data
    state_sample = np.load("results/Gridworld/transition-data/distance_current_states.npy")

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

        # Compute and normalize the features of all sample states
        features = Agent.phi(Agent.state.params, state_sample).out 
        norms = jnp.linalg.norm(features, axis=1)
        nonzero_mask = norms > 0  # Create a mask to filter out vectors with zero norms
        zero_norms = jnp.sum(~nonzero_mask)
        filtered_features = features[nonzero_mask]
        filtered_norms = norms[nonzero_mask]
        normalized_features = filtered_features / filtered_norms[:, None]
        
        # Compute pairwise dot products
        cosine_similarities = jnp.dot(normalized_features, normalized_features.T)
        upper_triangle_indices = jnp.triu_indices(cosine_similarities.shape[0], k=1)
        upper_triangle_values = cosine_similarities[upper_triangle_indices]

        # Convert to pairwise orthogonalities
        orthogonality = 1 - upper_triangle_values

        # Plot histogram
        ax.hist(orthogonality, bins=20, range=(0, 1), histtype='stepfilled', 
                color=COLORS[seed], alpha=0.3, label=f'seed={seed}, mean={round(jnp.mean(orthogonality).item(),5)}, # of empty features={zero_norms}')
        
        seed += 1
        
    ax.set_title(f'Distribution of Rep Orthogonality. {agent_label}. alpha={alpha}')
    ax.set_ylabel('Counts')
    ax.set_xlabel('Orthogonality')
    ax.legend()

    save(save_path=f'{path}/plots', plot_name=f'{agent_label}-{alpha}alpha')
    plt.show()

    print(f"Execution time: {time.time()-start} seconds.")

if __name__ == '__main__':
    main()