#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P0/gridworldpartial_drqn_blind_pretrain_hyper_sweep/DQN-ReLU-Gridworld.json
