#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-1-32-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-2-16-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-4-8-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-8-4-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-16-2-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-32-1-hyper-sweep.json