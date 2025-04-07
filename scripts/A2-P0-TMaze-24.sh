#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/t_maze_drqn_24_hyper_sweep/TMaze-DRQN-24-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/t_maze_drqn_24_use_all_steps_hyper_sweep/TMaze-DRQN-24-use-all-steps-hyper-sweep.json