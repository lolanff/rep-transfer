from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment.tools import parseCmdLineArgs

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()
    
    template = {
        "agent": None,
        "problem": "Gridworld",
        "total_steps": 100000,
        "episode_cutoff": 100,
        "metaParameters": {
            "experiment": {
                "load": False,
                "save": False,
                "seed_offset": 40000
            },
            "epsilon": 0.1,
            "target_refresh": 64,
            "buffer_type": "rnn_uniform",
            "buffer_size": 10000,
            "batch": 2,
            "n_step": 1,
            "update_freq": 1,
            "sequence_length": 16,
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
            "representation": None,
            "environment": {
                "goal_id": None,
            }
        }
    }
    
    RELU_REPRESENTATION = {
            "type": "MazeGRUNetReLU",
            "hidden": 32,
            "frozen": False
        }
    FTA_REPRESENTATION = {
            "type": "MazeGRUNetFTA",
            "hidden": 32,
            "eta": 0.2,
            "frozen": False
        }
    
    GOALS = [str(goal) for goal in range(0, 170 + 1, 5)]
    
    agents = {
        "DRQN-ReLU-Gridworld": {
            "representation": RELU_REPRESENTATION
        },
        "DRQN-FTA-Gridworld": {
            "representation": FTA_REPRESENTATION
        },
    }
    
    json_path_dir = Path(f"{path}/gridworld_drqn_2_16_scratch_hyper_sweep")
    json_path_dir.mkdir(exist_ok=True)
    
    json_paths = []

    for (agent, agent_config), goal in product(agents.items(), GOALS):
        config = deepcopy(template)
        config["agent"] = agent + "-" + goal
        config["metaParameters"]["environment"]["goal_id"] = goal
        config["metaParameters"]["representation"] = agent_config["representation"]
        json_path = json_path_dir / f'{config["agent"]}.json'
        json_paths.append(json_path)
        with open(json_path, "w") as f:
            gridworld_scratch_best = json.dump(config, f, indent=4)
        
    num_split = 32
    json_paths = np.array_split(json_paths, num_split)

    for i, file in enumerate(json_paths):
        script_name = f"{path}/../../../../scripts/A3/P3S/scratch-hyper-sweep-{i}.sh"
        os.makedirs(os.path.dirname(script_name), exist_ok=True)
        with open(script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n\n")
            for line in file:
                f.write(f"python scripts/local.py --runs 5 -e {line}\n")
        os.chmod(script_name, 0o755) 
        print(f"Generated: {script_name}")
    