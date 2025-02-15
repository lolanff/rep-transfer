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
scratch_path = Path(os.getcwd() + "/experiments/Gridworld/E1/P0/best-ScratchRelu")
scratch_path.mkdir(exist_ok=True)

old_best_alpha = {'-1': 0.0001, '104': 0.0001, '109': 0.0003, '114': 0.0001, '119': 0.0001, '124': 0.0001, '129': 0.0001, '134': 3e-05, '139': 0.0001, '14': 0.0001, '144': 0.0001, '149': 0.0001, '154': 0.0001, '159': 0.0001, '164': 0.0001, '169': 0.0001, '19': 0.0001, '24': 0.0001, '29': 0.0001, '34': 0.0001, '39': 0.0001, '4': 0.0001, '44': 0.0001, '49': 0.0003, '54': 0.0003, '59': 0.0001, '64': 0.0001, '69': 0.0003, '74': 0.0001, '79': 0.0003, '84': 0.0001, '89': 0.0001, '9': 0.0001, '94': 0.0001, '99': 0.0001}
best_alpha = {f'{int(key)+1}': old_best_alpha[key] for key in old_best_alpha.keys()}

scratch_json_paths = []

for goal, alpha in best_alpha.items():
    scratch = deepcopy(ReLU_template)
    scratch["agent"] += f"-goal-{goal}"
    scratch["metaParameters"]["environment"]["goal_id"] = goal
    scratch["metaParameters"]["experiment"]["seed_offset"] = 10000
    scratch["metaParameters"]["optimizer"]["alpha"] = alpha
    scratch_json_path = scratch_path / f"{scratch['agent']}.json"
    scratch_json_paths.append(scratch_json_path)
    with open(scratch_json_path, "w") as f:
        scratch_best = json.dump(scratch, f, indent=4)

script_name = os.getcwd() + "/scripts/best-ScratchRelu.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    for scratch_json_path in scratch_json_paths:
        f.write(f"python scripts/local.py --runs 5 -e experiments/Gridworld/E1/P0/{scratch_json_path} --cpus 16\n")
os.chmod(script_name, 0o755) 

# %%
# Generate configs and script for best Scratch(FTA)
scratch_fta_path = Path(os.getcwd() + "/experiments/Gridworld/E1/P0/best-ScratchFTA")
scratch_fta_path.mkdir(exist_ok=True)

fta_best_alpha = {'0': 0.0001, '10': 0.0003, '100': 0.0001, '105': 0.0001, '110': 0.0001, '115': 0.0001, '120': 0.0001, '125': 0.0001, '130': 0.0001, '135': 0.0003, '140': 0.0001, '145': 0.0001, '15': 0.0001, '150': 0.0001, '155': 0.0001, '160': 0.0003, '165': 0.0001, '170': 0.0001, '20': 0.0001, '25': 0.0003, '30': 0.0001, '35': 0.0003, '40': 0.0003, '45': 0.0001, '5': 0.0003, '50': 0.0001, '55': 0.0001, '60': 0.0001, '65': 0.0001, '70': 0.0003, '75': 0.0001, '80': 0.0001, '85': 0.0001, '90': 0.0001, '95': 0.0003}

scratch_fta_json_paths = []

for goal, alpha in fta_best_alpha.items():
    scratch = deepcopy(FTA_template)
    scratch["agent"] += f"-goal-{goal}"
    scratch["metaParameters"]["environment"]["goal_id"] = goal
    scratch["metaParameters"]["experiment"]["seed_offset"] = 10000
    scratch["metaParameters"]["optimizer"]["alpha"] = alpha
    scratch_fta_json_path = scratch_fta_path / f"{scratch['agent']}.json"
    scratch_fta_json_paths.append(scratch_fta_json_path)
    with open(scratch_fta_json_path, "w") as f:
        json.dump(scratch, f, indent=4)

script_name = os.getcwd() + "/scripts/best-ScratchFTA.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    for scratch_fta_json_path in scratch_fta_json_paths:
        f.write(f"python scripts/local.py --runs 5 -e experiments/Gridworld/E1/P0/{scratch_fta_json_path} --cpus 16\n")
os.chmod(script_name, 0o755) 


#%%
sample_goal = sorted([int(key) for key in best_alpha.keys()])
scratch_FTA_sweep = deepcopy(FTA_template)
scratch_FTA_sweep["agent"] += "-sweep"
scratch_FTA_sweep["metaParameters"]["environment"]["goal_id"] = [f'{id}' for id in sample_goal]
with open(os.getcwd() +  "/experiments/Gridworld/E1/P0/DQN-FTA-sweep-new.json", "w") as f:
    json.dump(scratch_FTA_sweep, f, indent=4)
