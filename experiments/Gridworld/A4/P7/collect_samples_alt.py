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
    save_path = f'{path}/sampled_buffer_alt.pkl.xz'
    
    with lzma.open(checkpoint_path, 'rb') as f:
        chk_storage = pickle.load(f)
    agent = chk_storage["a"]
    
    batch_seq_pair = []

    for i in tqdm(range(1000)):
        found_seq = None
        while True:
            batch = agent.buffer.sample_sequences(3)
            pos = batch.pos
            x = batch.x
            if np.allclose(pos[0, -1], pos[1, -1]) \
                    and not np.allclose(pos[0, -1], pos[2, -1]) \
                    and np.allclose(x[0, -1], x[2, -1]) \
                    and not np.allclose(pos[0, -1], [-1, -1]) \
                    and not np.allclose(pos[1, -1], [-1, -1]) \
                    and not np.allclose(pos[2, -1], [-1, -1]):
                found_seq = batch
                print(pos[0, -1], pos[1, -1], pos[2, -1])
                break
    
        batch_seq_pair.append(found_seq)

    with lzma.open(save_path, 'wb') as f:
        pickle.dump(batch_seq_pair, f)

if __name__ == '__main__':
    main()

