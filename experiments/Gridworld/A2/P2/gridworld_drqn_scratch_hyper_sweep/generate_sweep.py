# %%
from copy import deepcopy
import json
from itertools import product
from pathlib import Path
import numpy as np

template = {
    "agent": "DRQN-ReLU",
    "problem": "Gridworld",
    "total_steps": 100000,
    "episode_cutoff": 100,
    "early_saving": -1,
    "metaParameters": {
        "experiment": {
            "load": False,
            "seed_offset": 0
        },
        "epsilon": 0.1,
        "target_refresh": 64,
        "buffer_type": "rnn_uniform",
        "buffer_size": 10000,
        "batch": 32,
        "n_step": 1,
        "update_freq": 1,
        "sequence_length": 32,
        "train_use_all_steps": True,
        "normalizer": {
            "state": {
                "coeff": 255.0
            }
        },
        "optimizer": {
            "name": "ADAM",
            "alpha": [0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            "beta1": 0.9,
            "beta2": 0.999
        },
        "representation": {
            "type": "MazeGRUNetReLU",
            "hidden": 32,
            "frozen": False
        },
        "environment": {
            "goal_id": [
                "0",
                "5",
                "10",
                "15",
                "20",
                "25",
                "30",
                "35",
                "40",
                "45",
                "50",
                "55",
                "60",
                "65",
                "70",
                "75",
                "80",
                "85",
                "90",
                "95",
                "100",
                "105",
                "110",
                "115",
                "120",
                "125",
                "130",
                "135",
                "140",
                "145",
                "150",
                "155",
                "160",
                "165",
                "170"
            ]
        }
    }
}


#%%
gridworld_scratch_path = Path("scratch_sweep")
gridworld_scratch_path.mkdir(exist_ok=True)

#%%
agents = [("DQN-ReLU-scratch-", "results/Gridworld/A0/P1/pretrain/DQN-ReLU-A")]

#%%
goals = list(range(0, 173, 5))

#%%

gridworld_scratch_json_paths = []

for agent, goal in product(agents, goals):
    gridworld_scratch = deepcopy(template)
    gridworld_scratch["agent"] = agent[0] + str(goal)
    gridworld_scratch["metaParameters"]["environment"]["goal_id"] = goal
    gridworld_scratch["metaParameters"]["experiment"]["load"]["path"] = agent[1]
    gridworld_scratch_json_path = gridworld_scratch_path / f"{gridworld_scratch['agent']}.json"
    gridworld_scratch_json_paths.append(gridworld_scratch_json_path)
    with open(gridworld_scratch_json_path, "w") as f:
        gridworld_scratch_best = json.dump(gridworld_scratch, f, indent=4)

#%%
num_split = 8
gridworld_scratch_json_paths_segs = np.array_split(gridworld_scratch_json_paths, num_split)

# %%
for i, gridworld_scratch_json_paths in enumerate(gridworld_scratch_json_paths_segs):
    with open(f"../../../../scripts/A1-P0-{i}.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        for gridworld_scratch_json_path in gridworld_scratch_json_paths:
            f.write(f"python scripts/local.py --runs 5 -e experiments/sparse_feature_scratch/{gridworld_scratch_json_path} --cpus 16\n")

# %%
