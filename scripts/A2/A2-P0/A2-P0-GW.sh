#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/gridworld_dqn_hyper_sweep/Gridworld-DQN-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/gridworld_drqn_hyper_sweep/Gridworld-DRQN-hyper-sweep.json