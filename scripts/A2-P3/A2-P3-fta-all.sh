#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-1-32.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-2-16.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-4-8.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-8-4.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-16-2.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-32-1.json