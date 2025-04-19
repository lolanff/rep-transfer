#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-1-32.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-2-16.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-4-8.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-8-4.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-16-2.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P3/t_maze_drqn_fta_hyper_sweep/TMaze-DRQN-fta-32-1.json