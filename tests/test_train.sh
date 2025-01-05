#!/bin/sh

# exit script on error
set -e

# runs = 25 implies 5 random seeds if there are 5 sets of hyperparams
python scripts/local.py --runs 5 -e experiments/Gridworld/train/DQN-ReLU-A.json --cpus 25
python scripts/local.py --runs 5 -e experiments/Gridworld/train/DQN-ReLU-B.json --cpus 25
python experiments/Gridworld/learning_curve_train.py save

### old commands ###
#python src/main.py -e experiments/Gridworld/train/DQN-ReLU-A.json -i 0