#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/pretrain/DQN-ReLU-A.json --cpus 16

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-transfer-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-transfer-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-transfer-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-transfer-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-transfer-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-transfer-F.json --cpus 16
