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
        "problem": "GridworldPartial",
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
                "fov": 5
            }
        }
    }
    
    DQN_RELU_REPRESENTATION = {
            "type": "MazeNetReLU",
            "hidden": 32,
            "frozen": False
        }
    DQN_FTA_REPRESENTATION = {
            "type": "MazeNetFTA",
            "hidden": 32,
            "eta": 0.2,
            "frozen": False
        }
    DRQN_RELU_REPRESENTATION = {
            "type": "MazeGRUNetReLU",
            "hidden": 32,
            "frozen": False
        }
    DRQN_FTA_REPRESENTATION = {
            "type": "MazeGRUNetFTA",
            "hidden": 32,
            "eta": 0.2,
            "frozen": False
        }

    DQN_BATCH = 32
    DRQN_BATCH = 2
    DQN_BUFFER = "uniform"
    DRQN_BUFFER = "rnn_uniform"
    
    GOALS = [str(goal) for goal in range(0, 170 + 1, 5)]
    
    agents = {
        "DRQN-ReLU-Gridworld": {
            "representation": DRQN_RELU_REPRESENTATION,
            "batch": DRQN_BATCH,
            "buffer_type": DRQN_BUFFER
        },
        "DRQN-FTA-Gridworld": {
            "representation": DRQN_FTA_REPRESENTATION,
            "batch": DRQN_BATCH,
            "buffer_type": DRQN_BUFFER
        },
        "DQN-ReLU-Gridworld": {
            "representation": DQN_RELU_REPRESENTATION,
            "batch": DQN_BATCH,
            "buffer_type": DQN_BUFFER
        },
        "DQN-FTA-Gridworld": {
            "representation": DQN_FTA_REPRESENTATION,
            "batch": DQN_BATCH,
            "buffer_type": DQN_BUFFER
        }
    }
    
    json_path_dir = Path(f"{path}/gridworldpartial_scratch_hyper_sweep")
    json_path_dir.mkdir(exist_ok=True)
    
    json_paths = []

    for (agent, agent_config), goal in product(agents.items(), GOALS):
        config = deepcopy(template)
        config["agent"] = agent + "-" + goal
        config["metaParameters"]["environment"]["goal_id"] = goal
        config["metaParameters"]["representation"] = agent_config["representation"]
        config["metaParameters"]["buffer_type"] = agent_config["buffer_type"]
        config["metaParameters"]["batch"] = agent_config["batch"]
        json_path = json_path_dir / f'{config["agent"]}.json'
        json_paths.append(json_path)
        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)

    record_path = json_path_dir / 'record.txt'
    with open(record_path, "w") as f:
        print(" ".join(map(str, json_paths)), file=f)
    