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

from matplotlib.patches import Rectangle

from environments.GridworldPartial import GridHardRGBGoalPartial as Env
# from environments.GridworldGoal import GridHardRGBGoal as Env

setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

# Input params
chk_path = [
                'checkpoints/results/Gridworld/A4/P6/DQN-ReLU-GridworldPartial-0'
            ]

idxs = [0]    # enter the checkpoint indices

fov = 5

def normalize_state(x: np.ndarray, coeff=255.) -> np.ndarray:
    if coeff is not None:
        x = 2 * x / coeff - 1
    return x

def main():
    env = Env("0", fov)
    d, _, _ = env.state_dim
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for idx in idxs:
        state_dict = {}
        for i in range(d):
            for j in range(d):
                try:
                    state_dict[(i,j)] = {
                    'state': normalize_state(env.generate_state((i, j)))[None]
                }
                except:
                    pass

        for alg in chk_path:
            # Load the agent in the checkpoint
            with lzma.open(alg + f'/{idx}/chk.pkl.xz', 'rb') as f:
                chk_storage = pickle.load(f)
            agent = chk_storage["a"]

            for key, value in state_dict.items():
                phi = agent.phi(agent.state.params, value['state']).out
                q = agent.q(agent.state.params, phi)
                state_dict[key]['q'] = q

        # Plot policy
        fig, ax = plt.subplots()
        for i in range(d):
            for j in range(d):
                if (i, j) not in state_dict:
                    # Fill cells without a state in the dict in black
                    rect = Rectangle((j, d - 1 - i), 1, 1, color='black')
                    ax.add_patch(rect)
                else:
                    # Determine best action and draw an arrow
                    q_vals = np.array(state_dict[(i, j)]['q'])
                    a = int(np.argmax(q_vals))
                    dx, dy = actions[a]
                    ax.arrow(
                        j + 0.5,
                        d - 1 - i + 0.5,
                        dy * 0.3,
                        -dx * 0.3,
                        head_width=0.1,
                        head_length=0.1,
                        length_includes_head=True
                    )
        # Configure plot aesthetics
        ax.set_xlim(0, d)
        ax.set_ylim(0, d)
        ax.set_xticks(np.arange(0, d + 1))
        ax.set_yticks(np.arange(0, d + 1))
        ax.set_aspect('equal')
        ax.grid(True)
        
        save(
            save_path=f'{path}/plots',
            plot_name=f'policy_{idx}',
            f=fig,
            height_ratio=2/3,
        )
    
if __name__ == '__main__':
    main()