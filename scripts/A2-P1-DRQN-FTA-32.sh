#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A2/P1/gridworld_drqn_fta_hyper_sweep/Gridworld-DRQN-FTA-32-hyper-sweep.json --debug