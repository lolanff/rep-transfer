#!/bin/bash
set -e

python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-1-32-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-2-16-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-4-8-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-8-4-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-16-2-hyper-sweep.json
python scripts/local.py --runs 25 -e experiments/Gridworld/A2/P2/t_maze_drqn_fta_use_all_steps_hyper_sweep/TMaze-DRQN-fta-use-all-steps-32-1-hyper-sweep.json