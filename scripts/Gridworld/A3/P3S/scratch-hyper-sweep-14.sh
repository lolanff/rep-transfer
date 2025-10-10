#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P3S/gridworld_drqn_2_16_scratch_hyper_sweep/DRQN-ReLU-Gridworld-170.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P3S/gridworld_drqn_2_16_scratch_hyper_sweep/DRQN-FTA-Gridworld-0.json
