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
            "goal_id": ["-1"]
        }                
    }
}


#%%
gridworld_scratch_path = Path(os.getcwd() + "/experiments/Gridworld/E1/P0/best-ScratchRelu")
gridworld_scratch_path.mkdir(exist_ok=True)

#%%

old_best_alpha = {'-1': 0.0001, '104': 0.0001, '109': 0.0003, '114': 0.0001, '119': 0.0001, '124': 0.0001, '129': 0.0001, '134': 3e-05, '139': 0.0001, '14': 0.0001, '144': 0.0001, '149': 0.0001, '154': 0.0001, '159': 0.0001, '164': 0.0001, '169': 0.0001, '19': 0.0001, '24': 0.0001, '29': 0.0001, '34': 0.0001, '39': 0.0001, '4': 0.0001, '44': 0.0001, '49': 0.0003, '54': 0.0003, '59': 0.0001, '64': 0.0001, '69': 0.0003, '74': 0.0001, '79': 0.0003, '84': 0.0001, '89': 0.0001, '9': 0.0001, '94': 0.0001, '99': 0.0001}
best_alpha = {f'{int(key)+1}': old_best_alpha[key] for key in old_best_alpha.keys()}

gridworld_scratch_json_paths = []

for goal, alpha in best_alpha.items():
    gridworld_scratch = deepcopy(ReLU_template)
    gridworld_scratch["agent"] += f"-goal-{goal}"
    gridworld_scratch["metaParameters"]["environment"]["goal_id"] = goal
    gridworld_scratch["metaParameters"]["experiment"]["seed_offset"] = 10000
    gridworld_scratch["metaParameters"]["optimizer"]["alpha"] = alpha
    gridworld_scratch_json_path = gridworld_scratch_path / f"{gridworld_scratch['agent']}.json"
    gridworld_scratch_json_paths.append(gridworld_scratch_json_path)
    with open(gridworld_scratch_json_path, "w") as f:
        gridworld_scratch_best = json.dump(gridworld_scratch, f, indent=4)

# %%
script_name = os.getcwd() + "/scripts/best-ScratchRelu.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    for gridworld_scratch_json_path in gridworld_scratch_json_paths:
        f.write(f"python scripts/local.py --runs 5 -e experiments/Gridworld/E1/P0/{gridworld_scratch_json_path} --cpus 16 --gpu\n")
os.chmod(script_name, 0o755) 
# %%