#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/train/DQN-ReLU-A.json 
python scripts/local.py --runs 5 -e experiments/Gridworld/train/DQN-ReLU-B.json 
python experiments/Gridworld/learning_curve_train.py save

### old commands ###
#python src/main.py -e experiments/Gridworld/train/DQN-ReLU-A.json -i 0