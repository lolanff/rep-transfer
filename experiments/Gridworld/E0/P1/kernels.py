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

setDefaultConference('jmlr')

if __name__ == "__main__":
    idx = 3

    path, should_save, save_type = parseCmdLineArgs()

    exp = ExperimentModel.load('experiments/Gridworld/E0/P1/DQN-ReLU-A.json')
    chk = Checkpoint(exp, idx)
    with lzma.open(f'checkpoints/results/Gridworld/E0/P1/DQN-ReLU-A/{idx}/chk.pkl.xz', 'rb') as f:
        chk._storage = pickle.load(f)

    K = chk["a"].state.params["phi"]['conv']['w']

    # Plot each input channel
    for channel in range(3):
        channel_kernels = K[:, :, channel, :]  # Extract kernels for the current channel
        vmin, vmax = channel_kernels.min(), channel_kernels.max()  # Determine color scale

        # Create a figure and axes for 4x8 grid
        fig, axes = plt.subplots(4, 8)
        fig.suptitle(f"Channel {channel} Kernels (Min: {vmin:.2f}, Max: {vmax:.2f})", fontsize=12)

        for i in range(4):  # Rows
            for j in range(8):  # Columns
                kernel_index = i * 8 + j
                if kernel_index < channel_kernels.shape[-1]:  # Ensure within bounds
                    ax = axes[i, j]
                    kernel = channel_kernels[:, :, kernel_index]  # Extract single kernel
                    im = ax.imshow(kernel, cmap='binary', vmin=vmin, vmax=vmax)  # Plot with shared scale
                    ax.axis("off")
                else:
                    axes[i, j].axis("off")  # Hide unused axes

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Leave space for title
        save(
                save_path=f'{path}/plots',
                plot_name=f'{idx}/channel{channel}'
            )
        plt.clf()

        
    