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
                "load": {
                    "path": None,
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
                "save": False,
                "seed_offset": 20000
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
                "alpha": None,
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
            "frozen": True
        }
    FTA_REPRESENTATION = {
            "type": "MazeGRUNetFTA",
            "hidden": 32,
            "eta": 0.2,
            "frozen": True
        }
    RELU_ALPHA = [
            0.01,
            0.003,
            0.001,
            0.0003,
            0.0001
        ]
    FTA_ALPHA = [
            0.001,
            0.0003,
            0.0001,
            0.00003,
            0.00001
        ]
    
    GOALS = [str(goal) for goal in range(0, 170 + 1, 5)]
    
    agents = {
        "DRQN-ReLU-Gridworld": {
            "alpha": RELU_ALPHA,
            "path": "results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQN-ReLU-Gridworld",
            "representation": RELU_REPRESENTATION
        },
        "DRQNAux-ReLU-Gridworld": {
            "alpha": RELU_ALPHA,
            "path": "results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQNAux-ReLU-Gridworld",
            "representation": RELU_REPRESENTATION
        },
        "DRQN-FTA-Gridworld": {
            "alpha": FTA_ALPHA,
            "path": "results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQN-FTA-Gridworld",
            "representation": FTA_REPRESENTATION
        },
        "DRQNAux-FTA-Gridworld": {
            "alpha": FTA_ALPHA,
            "path": "results/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQNAux-FTA-Gridworld",
            "representation": FTA_REPRESENTATION
        },
    }
    
    json_path_dir = Path(f"{path}/gridworld_drqn_2_16_transfer_hyper_sweep")
    json_path_dir.mkdir(exist_ok=True)
    
    json_paths = []

    for (agent, agent_config), goal in product(agents.items(), GOALS):
        config = deepcopy(template)
        config["agent"] = agent + "-" + goal
        config["metaParameters"]["environment"]["goal_id"] = goal
        config["metaParameters"]["experiment"]["load"]["path"] = agent_config["path"]
        config["metaParameters"]["representation"] = agent_config["representation"]
        config["metaParameters"]["optimizer"]["alpha"] = agent_config["alpha"]
        json_path = json_path_dir / f'{config["agent"]}.json'
        json_paths.append(json_path)
        with open(json_path, "w") as f:
            gridworld_scratch_best = json.dump(config, f, indent=4)
        
    num_split = 4
    json_paths = np.array_split(json_paths, num_split)

    for i, file in enumerate(json_paths):
        script_name = f"{path}/../../../../scripts/A3/P3/transfer-hyper-sweep-{i}.sh"
        os.makedirs(os.path.dirname(script_name), exist_ok=True)
        with open(script_name, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n\n")
            for line in file:
                f.write(f"python scripts/local.py --runs 5 -e {line}\n")
        os.chmod(script_name, 0o755) 
        print(f"Generated: {script_name}")
    