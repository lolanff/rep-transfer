#!/bin/bash
set -e

python scripts/local.py --entry 'src/continuing_main.py' --runs 5 -e experiments/Forager/A1/P15/DRQN-7.json