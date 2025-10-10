#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P2/gridworldpartial_dqn_pretrain/DQN-FTA-GridworldPartial.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P2/gridworldpartial_dqn_pretrain/DQN-ReLU-GridworldPartial.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P2/gridworldpartial_drqn_2_16_pretrain/DRQN-FTA-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A4/P2/gridworldpartial_drqn_2_16_pretrain/DRQN-ReLU-Gridworld.json