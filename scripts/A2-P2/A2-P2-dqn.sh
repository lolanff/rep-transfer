#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_dqn_fta_hyper_sweep/TMaze-DQN-fta-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P2/t_maze_dqn_hyper_sweep/TMaze-DQN-hyper-sweep.json