#!/bin/sh

# exit script on error
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A0/P3/DQN-FTA-A.json --cpus 16