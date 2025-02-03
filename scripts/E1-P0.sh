#!/bin/sh

# exit script on error
set -e

#python scripts/local.py --runs 5 -e experiments/Gridworld/E1/P0/DQN-Relu.json 
python scripts/slurm.py --cluster clusters/cedar.json --runs 5 -e experiments/Gridworld/E1/P0/DQN-Relu.json 