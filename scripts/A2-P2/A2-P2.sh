#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_hyper_sweep/TMaze-DRQN-1-32-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_hyper_sweep/TMaze-DRQN-2-16-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_hyper_sweep/TMaze-DRQN-4-8-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_hyper_sweep/TMaze-DRQN-8-4-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_hyper_sweep/TMaze-DRQN-16-2-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_hyper_sweep/TMaze-DRQN-32-1-hyper-sweep.json