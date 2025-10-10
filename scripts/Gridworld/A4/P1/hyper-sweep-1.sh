#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P1/gridworldpartial_drqn_2_16_pretrain_hyper_sweep/DRQN-FTA-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P1/gridworldpartial_drqn_2_16_pretrain_hyper_sweep/DRQN-ReLU-Gridworld.json
