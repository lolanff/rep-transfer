#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P4/pretrain/DQN-FTA-A.json --cpus 32
python scripts/local.py --runs 5 -e experiments/Gridworld/A1/P2/transfer_sweep/DQN-FTA-transfer-sweep.json --cpus 32