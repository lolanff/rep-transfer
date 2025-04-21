#!/bin/bash
set -e

python scripts/local.py --runs 30 -e experiments/Gridworld/A2/P3/t_maze_drqn/TMaze-DRQN-1-32.json
python scripts/local.py --runs 30 -e experiments/Gridworld/A2/P3/t_maze_drqn/TMaze-DRQN-2-16.json
python scripts/local.py --runs 30 -e experiments/Gridworld/A2/P3/t_maze_drqn/TMaze-DRQN-4-8.json
python scripts/local.py --runs 30 -e experiments/Gridworld/A2/P3/t_maze_drqn/TMaze-DRQN-8-4.json
python scripts/local.py --runs 30 -e experiments/Gridworld/A2/P3/t_maze_drqn/TMaze-DRQN-16-2.json
python scripts/local.py --runs 30 -e experiments/Gridworld/A2/P3/t_maze_drqn/TMaze-DRQN-32-1.json