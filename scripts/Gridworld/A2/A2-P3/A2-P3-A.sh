#!/bin/bash
set -e

python scripts/local.py --runs 3000 -e experiments/Gridworld/A2/P3-A/t_maze_drqn_use_all_steps_1_32/TMaze-DRQN-use-all-steps-1-32.json
python experiments/Gridworld/A2/P3-A/collect_data_raw.py