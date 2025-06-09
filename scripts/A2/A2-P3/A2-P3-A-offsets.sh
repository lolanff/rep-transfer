#!/bin/bash
set -e

python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P3-A/t_maze_drqn_fta_use_all_steps_offsets/TMaze-DRQN-fta-use-all-steps-1-32-2000.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P3-A/t_maze_drqn_fta_use_all_steps_offsets/TMaze-DRQN-fta-use-all-steps-1-32-3000.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P3-A/t_maze_drqn_fta_use_all_steps_offsets/TMaze-DRQN-fta-use-all-steps-1-32-4000.json
python experiments/Gridworld/A2/P3-A/collect_data.py
