# %%
from copy import deepcopy
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import os

template = {
    "agent": "DQN-ReLU-transfer-A",
    "problem": "Gridworld",
    "total_steps": 100000,
    "episode_cutoff": 100,
    "metaParameters": {
        "experiment": {
            "load": {
                "path": "results/Gridworld/A0/P1/pretrain/DQN-ReLU-A",
                "config": {
                    "a": {
                        "buffer": False,
                        "state": {
                            "optim": False,
                            "params": {
                                "phi": True,
                                "q": False
                            },
                            "target_params": {
                                "phi": True,
                                "q": False
                            }
                        }
                    }
                }
            },
            "seed_offset": 20000
        },
        "epsilon": 0.1,
        "target_refresh": 64,
        "buffer_type": "uniform",
        "buffer_size": 10000,
        "batch": 32,
        "n_step": 1,
        "normalizer": {
            "state": {
                "coeff": 255.0
            }
        },
        "optimizer": {
            "name": "ADAM",
            "alpha": [0.01, 0.003, 0.001, 0.0003, 0.0001],
            "beta1": 0.9,
            "beta2": 0.999
        },
        "representation": {
            "type": "MazeNetReLU",
            "hidden": 32,
            "frozen": True
        },
        "environment": {
            "goal_id": "-1"
        }
    }
}

df_hyper_path = Path()
df = pd.read_csv(df_hyper_path / "../P0/hyperparameter_collector.csv", index_col=0)
#%%
gridworld_transfer_path = Path("transfer")
gridworld_transfer_path.mkdir(exist_ok=True)
#%%
agents = [("DQN-ReLU-transfer", "results/Gridworld/A0/P1/pretrain/DQN-ReLU-A"), (
        "DQN-ReLU-vf5-transfer", "results/Gridworld/A0/P1/pretrain/DQNAux-ReLU-A")]

#%%
goals = list(range(-1,172))

#%%

gridworld_transfer_json_paths = []

for agent, goal in product(agents, goals):
    gridworld_transfer = deepcopy(template)
    gridworld_transfer["agent"] = agent[0] + "-" + str(goal)
    gridworld_transfer["metaParameters"]["environment"]["goal_id"] = goal
    gridworld_transfer["metaParameters"]["experiment"]["load"]["path"] = agent[1]
    alpha = df[(df["Algorithm"] == agent[0]) & (df["Goal"] == goal)]["Value"].item()
    assert alpha is not None
    gridworld_transfer["metaParameters"]["optimizer"]["alpha"] = alpha
    gridworld_transfer_json_path = gridworld_transfer_path / f"{gridworld_transfer['agent']}.json"
    gridworld_transfer_json_paths.append(gridworld_transfer_json_path)
    with open(gridworld_transfer_json_path, "w") as f:
        gridworld_scratch_best = json.dump(gridworld_transfer, f, indent=4)

#%%
num_split = 8
gridworld_transfer_json_paths_segs = np.array_split(gridworld_transfer_json_paths, num_split)

# %%
for i, gridworld_transfer_json_paths in enumerate(gridworld_transfer_json_paths_segs):
    script_name = f"../../../../scripts/A1-P1/A1-P1-{i}.sh"
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        for gridworld_transfer_json_path in gridworld_transfer_json_paths:
            f.write(f"python scripts/local.py --runs 5 -e experiments/sparse_feature_scratch/{gridworld_transfer_json_path} --cpus 16\n")
    os.chmod(script_name, 0o755)  

# %%
