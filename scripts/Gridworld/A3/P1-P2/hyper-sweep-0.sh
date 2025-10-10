#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P1/gridworld_drqn_2_16_pretrain_hyper_sweep/DRQN-FTA-Gridworld.json
