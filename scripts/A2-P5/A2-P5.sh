#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P5/gridworld_dqn_fta/Gridworld-DQN-fta.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P5/gridworld_dqn/Gridworld-DQN.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P5/gridworld_drqn_fta/Gridworld-DRQN-fta.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P5/gridworld_drqn/Gridworld-DRQN.json