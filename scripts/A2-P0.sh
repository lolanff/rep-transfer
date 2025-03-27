#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P0/memory_drqn_hyper_sweep/Memory-DRQN-hyper-sweep.json --debug