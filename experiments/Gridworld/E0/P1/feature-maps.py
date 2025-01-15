import os
import sys
sys.path.append(os.getcwd() + '/src')

from environments.GridworldGoal import GridHardRGBGoal as Env
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
import pickle
import lzma
from experiment.tools import parseCmdLineArgs
import jax
import jax.numpy as jnp

setDefaultConference('jmlr')

idx = 0
agent_coord = [7, 7] # [14,0] is bottom left. [0,14] is top right.

path, should_save, save_type = parseCmdLineArgs()

def main(): 
    exp = ExperimentModel.load(f'{path}/DQN-ReLU-A.json')
    chk = Checkpoint(exp, idx)
    with lzma.open(f'checkpoints/results/Gridworld/E0/P1/DQN-ReLU-A/{idx}/chk.pkl.xz', 'rb') as f:
        chk._storage = pickle.load(f)

    gridworld_A = Env('A', seed=0)
    state = gridworld_A.generate_state(agent_coord)

    # Plot the gridworld state
    plt.figure(figsize=(6, 6))
    plt.imshow(state, origin='upper')  
    plt.title("Agent View")
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.xticks(range(15))
    plt.yticks(range(15))
    plt.show()

    # Retrive weights and biases in trained phi network
    W0 = chk["a"].state.target_params["phi"]['conv']['w']  # First conv layer weights
    B0 = chk["a"].state.target_params["phi"]['conv']['b']  # First conv layer biases
    W1 = chk["a"].state.target_params["phi"]['conv_1']['w']  # Second conv layer weights
    B1 = chk["a"].state.target_params["phi"]['conv_1']['b']  # Second conv layer biases

    # Compute feature maps
    state = jnp.expand_dims(state, axis=0) 
    feature_maps_1 = convolve_and_activate(state, W0, B0)
    feature_maps_2 = convolve_and_activate(feature_maps_1, W1, B1)

    # Visualize feature maps
    visualize_feature_maps(feature_maps_1[0], 0)
    visualize_feature_maps(feature_maps_2[0], 1)

def convolve_and_activate(input_tensor, W, B, stride=1, padding='SAME'):
    # Perform convolution (use JAX's lax for a low-level convolution operation)
    conv = jax.lax.conv_general_dilated(
        lhs=input_tensor,
        rhs=W,
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')  # Input format, kernel format, output format
    )
    # Add bias and apply ReLU activation
    return jax.nn.relu(conv + B)

# Visualize feature maps
def visualize_feature_maps(feature_maps, layer_idx):
    num_maps = feature_maps.shape[-1]
    grid_rows = 4
    grid_cols = int(num_maps/grid_rows)

    fig, axes = plt.subplots(grid_rows, grid_cols)
    fig.suptitle(f"Feature Maps After #{layer_idx} Conv Layer. Agent at {agent_coord}", fontsize=12)

    vmin, vmax = feature_maps.min(), feature_maps.max()
    for i in range(grid_rows * grid_cols):
        ax = axes[i // grid_cols, i % grid_cols]
        if i < num_maps:
            im = ax.imshow(feature_maps[..., i], cmap='binary', vmin=vmin, vmax=vmax)
            ax.axis("on")
            ax.set_xticks([])  
            ax.set_yticks([]) 
        else:
            ax.axis("off")
    fig.colorbar(im, orientation='horizontal')
    plt.tight_layout()
    save(save_path=f'{path}/plots', plot_name=f'{idx}/{agent_coord[0]}-{agent_coord[1]}_conv{layer_idx}')
    plt.clf()
   
if __name__ == '__main__':
    main()