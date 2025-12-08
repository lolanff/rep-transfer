import os
import sys
sys.path.append(os.getcwd())

from experiment import ExperimentModel
from experiment.tools import parseCmdLineArgs
from utils.checkpoint import Checkpoint
import pickle
import lzma
import argparse
import jax.numpy as jnp
import numpy as np
import time
from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
    parser.add_argument('--save_path', type=str, default='./')

    args = parser.parse_args()
    
    path, should_save, save_type = parseCmdLineArgs()

    exp = ExperimentModel.load(args.exp)
    
    exp_path_base = args.exp.removesuffix('.json')
    checkpoint_path_base = exp_path_base.replace('experiments', 'checkpoints/results')
    
    for idx in args.idxs:
        checkpoint_path = f'{checkpoint_path_base.replace("_sample", "")}/{idx}/chk.pkl.xz'
        save_path = f'{exp_path_base}/{idx}/sampled_buffer_alt.pkl.xz'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with lzma.open(checkpoint_path, 'rb') as f:
            chk_storage = pickle.load(f)
        agent = chk_storage["a"]

        batch_seq_pair = []

        for i in tqdm(range(100)):
            found_seq = None
            while True:
                batch = agent.buffer.sample_sequences(3, sequence_length=100)
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

