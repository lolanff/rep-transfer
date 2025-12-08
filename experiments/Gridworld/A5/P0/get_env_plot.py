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

import numpy as np
import time
import pandas as pd

from matplotlib.patches import Rectangle

from environments.GridworldPartial import GridHardRGBGoalPartial as Env
# from environments.GridworldGoal import GridHardRGBGoal as Env

setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

fov = 5

def normalize_state(x: np.ndarray, coeff=255.) -> np.ndarray:
    if coeff is not None:
        x = 2 * x / coeff - 1
    return x

def main():
    env = Env("0", fov)
    xd, yd, _ = env.state_dim

    state_dict = {}
    for i in range(xd):
        for j in range(yd):
            try:
                state_dict[(i,j)] = {
                'state': normalize_state(env.generate_state((i, j)))[None]
            }
            except:
                pass

    # Plot policy
    fig, ax = plt.subplots()

    for i in range(xd):
        for j in range(yd):
            if (i, j) not in state_dict:
                # Fill cells without a state in the dict in black
                rect = Rectangle((j, i), 1, 1, color='black')
                ax.add_patch(rect)
            else:
                pass
    # Configure plot aesthetics
    ax.set_xlim(0, yd)
    ax.set_ylim(0, xd)
    ax.set_xticks(np.arange(0, yd + 1))
    ax.set_yticks(np.arange(0, xd + 1))
    ax.invert_yaxis()
    ax.grid(True)
    
    save(
        save_path=f'{path}/plots',
        plot_name=f'env',
        f=fig,
    )
    
if __name__ == '__main__':
    main()