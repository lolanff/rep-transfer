#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P1/gridworld_drqn_hyper_sweep/Gridworld-DRQN-hyper-sweep.json --debug