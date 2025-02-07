# %%
from copy import deepcopy
import json
from itertools import product
from pathlib import Path
import numpy as np

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


#%%
gridworld_transfer_path = Path("transfer_sweep")
gridworld_transfer_path.mkdir(exist_ok=True)

#%%
agents = [("DQN-ReLU-transfer-", "results/Gridworld/A0/P1/pretrain/DQN-ReLU-A"), (
        "DQN-ReLU-vf5-transfer-", "results/Gridworld/A0/P1/pretrain/DQNAux-ReLU-A")]

#%%
goals = list(range(-1,172))
goals

#%%

gridworld_transfer_json_paths = []

for agent, goal in product(agents, goals):
    gridworld_transfer = deepcopy(template)
    gridworld_transfer["agent"] = agent[0] + str(goal)
    gridworld_transfer["metaParameters"]["environment"]["goal_id"] = goal
    gridworld_transfer["metaParameters"]["experiment"]["load"]["path"] = agent[1]
    gridworld_transfer_json_path = gridworld_transfer_path / f"{gridworld_transfer['agent']}.json"
    gridworld_transfer_json_paths.append(gridworld_transfer_json_path)
    with open(gridworld_transfer_json_path, "w") as f:
        gridworld_scratch_best = json.dump(gridworld_transfer, f, indent=4)

#%%
num_split = 8
gridworld_transfer_json_paths_segs = np.array_split(gridworld_transfer_json_paths, num_split)

# %%
for i, gridworld_transfer_json_paths in enumerate(gridworld_transfer_json_paths_segs):
    with open(f"../../../../scripts/A1-P0-{i}.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        for gridworld_transfer_json_path in gridworld_transfer_json_paths:
            f.write(f"python scripts/local.py --runs 5 -e experiments/sparse_feature_scratch/{gridworld_transfer_json_path} --cpus 16\n")

# %%
