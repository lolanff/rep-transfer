#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P1/gridworld_drqn_2_16_pretrain_hyper_sweep/DRQN-FTA-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P1/gridworld_drqn_2_16_pretrain_hyper_sweep/DRQN-ReLU-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P1/gridworld_drqn_2_16_pretrain_hyper_sweep/DRQNAux-FTA-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P1/gridworld_drqn_2_16_pretrain_hyper_sweep/DRQNAux-ReLU-Gridworld.json

python experiments/Gridworld/A3/P1/collect_hyper.py

python experiments/Gridworld/A3/P2/generate_best_hyper.py

python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQN-FTA-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQN-ReLU-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQNAux-FTA-Gridworld.json
python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P2/gridworld_drqn_2_16_pretrain/DRQNAux-ReLU-Gridworld.json