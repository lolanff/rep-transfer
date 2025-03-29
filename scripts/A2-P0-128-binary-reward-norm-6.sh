#!/bin/bash
set -e

python scripts/local.py --runs 1 -e experiments/Gridworld/A2/P0/memory_drqn_hyper_sweep/Memory-DRQN-hyper-sweep-128-binary-reward-norm-6.json --debug