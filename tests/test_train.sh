#!/bin/sh

# exit script on error
set -e

python src/main.py -e experiments/Gridworld/train/DQN-ReLU-A.json -i 0
python src/main.py -e experiments/Gridworld/train/DQN-ReLU-B.json -i 0
python experiments/Gridworld/learning_curve_train.py save