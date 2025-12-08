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

from environments.GridworldPartial import GridHardRGBGoalPartial as Env

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()
    
    template = {
        "agent": None,
        "problem": "GridworldPartial",
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
                "fov": 5
            }
        }
    }
    
    DQN_RELU_REPRESENTATION = {
            "type": "MazeARNet",
            "hidden": 32,
            "actions": 4,
            "frozen": True
        }
    DQN_FTA_REPRESENTATION = {
            "type": "MazeARNet",
            "hidden": 32,
            "actions": 4,
            "use_fta_with_eta": 0.2,
            "frozen": True
        }
    DRQN_RELU_REPRESENTATION = {
            "type": "MazeARGRUNet",
            "hidden": 32,
            "actions": 4,
            "frozen": True
        }
    DRQN_FTA_REPRESENTATION = {
            "type": "MazeARGRUNet",
            "hidden": 32,
            "actions": 4,
            "use_fta_with_eta": 0.2,
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
    DQN_BATCH = 32
    DRQN_BATCH = 2
    DQN_SEQUENCE_LENGTH = 1
    DRQN_SEQUENCE_LENGTH = 16
    DQN_BUFFER = "rnn_uniform"
    DRQN_BUFFER = "rnn_uniform"
    
    GOALS = [str(goal) for goal in range(0, Env(0, 0).get_num_states() + 1, 10)]
    
    agents = {
        "ARDRQN-ReLU-GridworldPartial": {
            "alpha": RELU_ALPHA,
            "path": "results/Gridworld/A5/P2/gridworldpartial_pretrain/ARDRQN-ReLU-GridworldPartial",
            "representation": DRQN_RELU_REPRESENTATION,
            "batch": DRQN_BATCH,
            "buffer_type": DRQN_BUFFER,
            "sequence_length": DRQN_SEQUENCE_LENGTH
        },
        "ARDRQN-FTA-GridworldPartial": {
            "alpha": FTA_ALPHA,
            "path": "results/Gridworld/A5/P2/gridworldpartial_pretrain/ARDRQN-FTA-GridworldPartial",
            "representation": DRQN_FTA_REPRESENTATION,
            "batch": DRQN_BATCH,
            "buffer_type": DRQN_BUFFER,
            "sequence_length": DRQN_SEQUENCE_LENGTH
        },
        "ARDQN-ReLU-GridworldPartial": {
            "alpha": RELU_ALPHA,
            "path": "results/Gridworld/A5/P2/gridworldpartial_pretrain/ARDQN-ReLU-GridworldPartial",
            "representation": DQN_RELU_REPRESENTATION,
            "batch": DQN_BATCH,
            "buffer_type": DQN_BUFFER,
            "sequence_length": DQN_SEQUENCE_LENGTH
        },
        "ARDQN-FTA-GridworldPartial": {
            "alpha": FTA_ALPHA,
            "path": "results/Gridworld/A5/P2/gridworldpartial_pretrain/ARDQN-FTA-GridworldPartial",
            "representation": DQN_FTA_REPRESENTATION,
            "batch": DQN_BATCH,
            "buffer_type": DQN_BUFFER,
            "sequence_length": DQN_SEQUENCE_LENGTH
        }
    }
    
    json_path_dir = Path(f"{path}/gridworldpartial_transfer_hyper_sweep")
    json_path_dir.mkdir(exist_ok=True)
    
    json_paths = []

    for (agent, agent_config), goal in product(agents.items(), GOALS):
        config = deepcopy(template)
        config["agent"] = agent + "-" + goal
        config["metaParameters"]["environment"]["goal_id"] = goal
        config["metaParameters"]["experiment"]["load"]["path"] = agent_config["path"]
        config["metaParameters"]["representation"] = agent_config["representation"]
        config["metaParameters"]["optimizer"]["alpha"] = agent_config["alpha"]
        config["metaParameters"]["buffer_type"] = agent_config["buffer_type"]
        config["metaParameters"]["batch"] = agent_config["batch"]
        config["metaParameters"]["sequence_length"] = agent_config["sequence_length"]
        json_path = json_path_dir / f'{config["agent"]}.json'
        json_paths.append(json_path)
        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)

    print(" ".join(map(str, json_paths)))
    