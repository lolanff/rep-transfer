#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/t_maze_drqn_1_hyper_sweep/TMaze-DRQN-1-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/t_maze_drqn_simple_12_hyper_sweep/TMaze-DRQN-simple-12-hyper-sweep.json