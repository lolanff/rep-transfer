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