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

setDefaultConference('jmlr')

def main():
    path, should_save, save_type = parseCmdLineArgs()
    
    checkpoint_path = 'checkpoints/results/Gridworld/A4/P5/random/DRQN-GridworldPartial-Random/0/chk.pkl.xz'
    save_path = f'{path}/sampled_buffer.pkl.xz'
    
    with lzma.open(checkpoint_path, 'rb') as f:
        chk_storage = pickle.load(f)
    agent = chk_storage["a"]

    batch = agent.buffer.sample_sequences(1000)

    with lzma.open(save_path, 'wb') as f:
        pickle.dump(batch, f)

if __name__ == '__main__':
    main()
