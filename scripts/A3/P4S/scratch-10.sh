#!/bin/bash
set -e

python scripts/local.py --runs 5 -e experiments/Gridworld/A3/P4S/gridworld_drqn_2_16_scratch/DRQN-ReLU-Gridworld-110.json
