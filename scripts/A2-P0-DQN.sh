#!/bin/bash
set -e

python scripts/local.py --runs 1 -e experiments/Gridworld/A2/P0/memory_dqn_hyper_sweep/Memory-DQN-hyper-sweep.json --cpus 16