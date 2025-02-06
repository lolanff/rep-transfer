#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/pretrain/DQNAux-ReLU-A.json --cpus 16

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-vf5-transfer-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-vf5-transfer-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-vf5-transfer-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-vf5-transfer-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-vf5-transfer-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/transfer/DQN-ReLU-vf5-transfer-F.json --cpus 16

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/pretrain/DQNAux-ReLU-A.json --cpus 16

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/transfer/DQN-ReLU-vf5-transfer-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/transfer/DQN-ReLU-vf5-transfer-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/transfer/DQN-ReLU-vf5-transfer-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/transfer/DQN-ReLU-vf5-transfer-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/transfer/DQN-ReLU-vf5-transfer-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1-A/transfer/DQN-ReLU-vf5-transfer-F.json --cpus 16