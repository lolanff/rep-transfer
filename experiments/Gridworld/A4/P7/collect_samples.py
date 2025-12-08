import os
import sys
sys.path.append(os.getcwd() + '/src')

from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
import pickle
import lzma
from experiment.tools import parseCmdLineArgs
import jax.numpy as jnp
import numpy as np
import time
from tqdm import tqdm
import pandas as pd


def main():
    path, should_save, save_type = parseCmdLineArgs()
    
    checkpoint_path = 'checkpoints/results/Gridworld/A4/P7/random/DRQN-GridworldPartial-Random/0/chk.pkl.xz'
    save_path = f'{path}/sampled_buffer.pkl.xz'
    
    with lzma.open(checkpoint_path, 'rb') as f:
        chk_storage = pickle.load(f)
    agent = chk_storage["a"]
    
    batch_seq_pair = []

    for i in tqdm(range(1000)):
        found_seq = None
        while True:
            batch = agent.buffer.sample_sequences(2)
            pos = batch.pos
            if np.allclose(pos[0, -1], pos[1, -1]):
                found_seq = batch
                break
    
        batch_seq_pair.append(found_seq)
    
    batch = agent.buffer.sample_sequences(1000)
    
    d = {
        'seq_pairs': batch_seq_pair,
        'indep_seqs': batch
    }
    with lzma.open(save_path, 'wb') as f:
        pickle.dump(d, f)

if __name__ == '__main__':
    main()
