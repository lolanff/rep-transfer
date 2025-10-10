#!/bin/bash
set -e

python scripts/local.py --entry 'src/continuing_main.py' --runs 5 -e experiments/Forager/A1/P12/DRQN-3.json