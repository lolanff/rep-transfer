#!/bin/bash
set -e

python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P3/t_maze_dqn/TMaze-DQN.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P3/t_maze_dqn_fta/TMaze-DQN-fta.json