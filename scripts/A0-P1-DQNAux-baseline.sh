#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-vf5-scratch-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-vf5-scratch-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-vf5-scratch-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-vf5-scratch-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-vf5-scratch-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-vf5-scratch-F.json --cpus 16
