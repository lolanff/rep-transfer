#%%
from copy import deepcopy
import json
from itertools import product
from pathlib import Path
import os 

ReLU_template = {
    "agent": "DQN-Relu",
    "problem": "Gridworld",
    "total_steps": 100000,
    "episode_cutoff": 100,
    "early_saving": -1,
    "metaParameters": {
        "experiment": {
            "load": False,
            "save": False,
            "seed_offset": 0
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
            "alpha": [0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            "beta1": 0.9,
            "beta2": 0.999
        },
        "representation": {
            "type": "MazeNetReLU",
            "hidden": 32,
            "frozen": False
        },
        "environment": {
            "goal_id": "0"
        }                
    }
}

FTA_template = {
    "agent": "DQN-FTA",
    "problem": "Gridworld",
    "total_steps": 100000,
    "episode_cutoff": 100,
    "early_saving": -1,
    "metaParameters": {
        "experiment": {
            "load": False,
            "save": False,
            "seed_offset": 0
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
            "alpha": [0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            "beta1": 0.9,
            "beta2": 0.999
        },
        "representation": {
            "type": "MazeNetFTA",
            "hidden": 32,
            "eta": 0.2,
            "frozen": False
        },
        "environment": {
            "goal_id": "0"
        }
    }
}

#%%
# Generate configs and script for best Scratch(Relu)
config_folder = "experiments/Gridworld/E1/P0/best-ScratchRelu"
config_folder_path = Path(os.getcwd() + "/" + config_folder)
config_folder_path.mkdir(exist_ok=True)

old_best_alpha = {'-1': 0.0001, '104': 0.0001, '109': 0.0003, '114': 0.0001, '119': 0.0001, '124': 0.0001, '129': 0.0001, '134': 3e-05, '139': 0.0001, '14': 0.0001, '144': 0.0001, '149': 0.0001, '154': 0.0001, '159': 0.0001, '164': 0.0001, '169': 0.0001, '19': 0.0001, '24': 0.0001, '29': 0.0001, '34': 0.0001, '39': 0.0001, '4': 0.0001, '44': 0.0001, '49': 0.0003, '54': 0.0003, '59': 0.0001, '64': 0.0001, '69': 0.0003, '74': 0.0001, '79': 0.0003, '84': 0.0001, '89': 0.0001, '9': 0.0001, '94': 0.0001, '99': 0.0001}
best_alpha = {f'{int(key)+1}': old_best_alpha[key] for key in old_best_alpha.keys()}

config_paths = []
for goal, alpha in best_alpha.items():
    config = deepcopy(ReLU_template)
    config["agent"] += f"-goal-{goal}"
    config["metaParameters"]["environment"]["goal_id"] = goal
    config["metaParameters"]["experiment"]["seed_offset"] = 10000
    config["metaParameters"]["optimizer"]["alpha"] = alpha
    config_path = config_folder_path / f"{config['agent']}.json"
    config_paths.append(config_folder + f"/{config['agent']}.json")
    with open(config_path, "w") as f:
        scratch_best = json.dump(config, f, indent=4)

script_name = os.getcwd() + "/scripts/best-ScratchRelu.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    for config_path in config_paths:
        f.write(f"python scripts/local.py --runs 5 -e {config_path} --cpus 16\n")
os.chmod(script_name, 0o755) 

# %%
# Generate configs and script for best Scratch(FTA)
config_folder = "experiments/Gridworld/E1/P0/best-ScratchFTA"
config_folder_path = Path(os.getcwd() + "/" + config_folder)
config_folder_path.mkdir(exist_ok=True)

best_alpha = {'0': 0.0001, '10': 0.0003, '100': 0.0001, '105': 0.0001, '110': 0.0001, '115': 0.0001, '120': 0.0001, '125': 0.0001, '130': 0.0001, '135': 0.0003, '140': 0.0001, '145': 0.0001, '15': 0.0001, '150': 0.0001, '155': 0.0001, '160': 0.0003, '165': 0.0001, '170': 0.0001, '20': 0.0001, '25': 0.0003, '30': 0.0001, '35': 0.0003, '40': 0.0003, '45': 0.0001, '5': 0.0003, '50': 0.0001, '55': 0.0001, '60': 0.0001, '65': 0.0001, '70': 0.0003, '75': 0.0001, '80': 0.0001, '85': 0.0001, '90': 0.0001, '95': 0.0003}

config_paths = []
for goal, alpha in best_alpha.items():
    config = deepcopy(FTA_template)
    config["agent"] += f"-goal-{goal}"
    config["metaParameters"]["environment"]["goal_id"] = goal
    config["metaParameters"]["experiment"]["seed_offset"] = 10000
    config["metaParameters"]["optimizer"]["alpha"] = alpha
    config_path = config_folder_path / f"{config['agent']}.json"
    config_paths.append(config_folder + f"/{config['agent']}.json")
    with open(config_path, "w") as f:
        scratch_best = json.dump(config, f, indent=4)

script_name = os.getcwd() + "/scripts/best-ScratchFTA.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    for config_path in config_paths:
        f.write(f"python scripts/local.py --runs 5 -e {config_path} --cpus 16\n")
os.chmod(script_name, 0o755) 