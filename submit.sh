#!/bin/bash
#SBATCH --time=00-02:59
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=1
#SBATCH --account=def-whitem
#SBATCH --job-name=scheduler
#SBATCH --output=scheduler.out

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif python scripts/slurm.py --cluster clusters/cedar.json --runs 5 -e experiments/Gridworld/E1/P0/DQN-Relu.json 