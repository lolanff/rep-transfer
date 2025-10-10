#!/bin/bash
set -e

python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P4/gridworld_dqn_fta_hyper_sweep/Gridworld-DQN-fta-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P4/gridworld_dqn_hyper_sweep/Gridworld-DQN-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P4/gridworld_drqn_fta_hyper_sweep/Gridworld-DRQN-fta-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P4/gridworld_drqn_hyper_sweep/Gridworld-DRQN-hyper-sweep.json