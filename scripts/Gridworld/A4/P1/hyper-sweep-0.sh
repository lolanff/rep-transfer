#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P1/gridworldpartial_dqn_pretrain_hyper_sweep/DQN-FTA-GridworldPartial.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P1/gridworldpartial_dqn_pretrain_hyper_sweep/DQN-ReLU-GridworldPartial.json
