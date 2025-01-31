#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-scratch-A.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-scratch-B.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-scratch-C.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-scratch-D.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-scratch-E.json --cpus 16
python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P1/scratch/DQNAux-ReLU-scratch-F.json --cpus 16
