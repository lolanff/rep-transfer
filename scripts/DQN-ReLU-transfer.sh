#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/E1/frozen/DQN-ReLU-A-rep0.json 
#python scripts/local.py --runs 5 -e experiments/Gridworld/E1/transfer/DQN-ReLUL-A-rep0.json 
