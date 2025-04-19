#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/memory_dqn_hyper_sweep/Memory-DQN-hyper-sweep.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/memory_dqn_hyper_sweep/Memory-DQN-hyper-sweep-norm-6.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/memory_dqn_hyper_sweep/Memory-DQN-hyper-sweep-binary-reward-norm-6.json