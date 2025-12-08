import os
import sys
sys.path.append(os.getcwd())

import pickle
import lzma
import argparse
import jax.numpy as jnp
import numpy as np
import pandas as pd

from experiment import ExperimentModel
from experiment.tools import parseCmdLineArgs


def add_entry(df, algorithm, seed, sa):
    new_entry = pd.DataFrame([[algorithm, seed, sa]], columns=df.columns)
    print(new_entry)
    return pd.concat([df, new_entry], ignore_index=True)


def state_awareness(phi_t1, phi_t2, phi_t3):
    # Compute feature distances
    dist_same = jnp.linalg.norm(phi_t1 - phi_t2, axis=1)
    dist_other = jnp.linalg.norm(phi_t1 - phi_t3, axis=1)

    return (np.sum(dist_other) - np.sum(dist_same)) / (np.sum(dist_other) + 1e-10).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', type=str, required=True)
    parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
    parser.add_argument('-c', '--chk', nargs='+', dest='chk_paths', required=True,
                        help='One or more checkpoint base paths (e.g. checkpoints/results/.../ALG')
    parser.add_argument('--save_path', type=str, default='./')

    args = parser.parse_args()

    # parse common slurm / plotting args
    path, should_save, save_type = parseCmdLineArgs()

    # load experiment to compute paths
    exp = ExperimentModel.load(args.exp)
    exp_path_base = args.exp.removesuffix('.json')

    df = pd.DataFrame(columns=['algorithm', 'seed', 'State Awareness'])

    for i in args.idxs:
        # load the sampled buffer produced by collect_samples_alt.py
        batch_path = f'{exp_path_base}/{i}/sampled_buffer_alt.pkl.xz'
        if not os.path.exists(batch_path):
            print(f'Warning: sampled buffer not found at {batch_path}, skipping index {i}')
            continue

        with lzma.open(batch_path, 'rb') as f:
            batch = pickle.load(f)

        traj = []
        for seq_pair in batch:
            traj.append(seq_pair.x)
        traj = jnp.array(traj)

        for alg in args.chk_paths:
            chk_file = alg + f'/{i}/chk.pkl.xz'
            if not os.path.exists(chk_file):
                print(f'Warning: checkpoint not found at {chk_file}, skipping')
                continue

            with lzma.open(chk_file, 'rb') as f:
                chk_storage = pickle.load(f)
            agent = chk_storage["a"]

            if 'drqn' in alg:
                phi_t1 = agent.phi(agent.state.params, traj[:, 0])[0][:, -1]
                phi_t2 = agent.phi(agent.state.params, traj[:, 1])[0][:, -1]
                phi_t3 = agent.phi(agent.state.params, traj[:, 2])[0][:, -1]
            else:
                phi_t1 = agent.phi(agent.state.params, traj[:, 0, -1]).out
                phi_t2 = agent.phi(agent.state.params, traj[:, 1, -1]).out
                phi_t3 = agent.phi(agent.state.params, traj[:, 2, -1]).out

            sa = state_awareness(phi_t1, phi_t2, phi_t3)
            print(f'State awareness of Rep #{i} for {alg} is {sa}.')

            df = add_entry(df, os.path.basename(alg), i, sa)

    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}/results_alt.csv")


if __name__ == '__main__':
    main()