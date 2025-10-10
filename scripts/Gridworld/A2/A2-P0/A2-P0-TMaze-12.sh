#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/t_maze_drqn_12_hyper_sweep/TMaze-DRQN-12-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/t_maze_drqn_12_use_all_steps_hyper_sweep/TMaze-DRQN-12-use-all-steps-hyper-sweep.json