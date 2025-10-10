#!/bin/bash
set -e

python scripts/local.py --entry 'src/continuing_main.py' --runs 5 -e experiments/Forager/A2/P1/DRQN-9.json