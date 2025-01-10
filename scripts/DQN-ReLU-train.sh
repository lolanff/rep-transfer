#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/E1/train/DQN-ReLU-A.json 
#python scripts/local.py --runs 5 -e experiments/Gridworld/E1/train/DQN-ReLUL-A.json 
