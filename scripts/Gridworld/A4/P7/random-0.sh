#!/bin/bash
set -e

python scripts/local.py --runs 1 -e experiments/Gridworld/A4/P7/random/DRQN-GridworldPartial-Random.json
