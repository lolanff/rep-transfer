#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-transfer-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-transfer-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-transfer-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-transfer-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-transfer-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-transfer-F.json --cpus 16

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-vf5-transfer-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-vf5-transfer-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-vf5-transfer-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-vf5-transfer-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-vf5-transfer-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer/DQN-ReLU-vf5-transfer-F.json --cpus 16

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer_no_early_saving/DQN-ReLU-transfer-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer_no_early_saving/DQN-ReLU-transfer-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer_no_early_saving/DQN-ReLU-transfer-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer_no_early_saving/DQN-ReLU-transfer-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer_no_early_saving/DQN-ReLU-transfer-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P2/transfer_no_early_saving/DQN-ReLU-transfer-F.json --cpus 16