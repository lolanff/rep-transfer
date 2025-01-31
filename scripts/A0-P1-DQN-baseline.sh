#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQN-ReLU-scratch-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQN-ReLU-scratch-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQN-ReLU-scratch-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQN-ReLU-scratch-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQN-ReLU-scratch-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQN-ReLU-scratch-F.json --cpus 16
