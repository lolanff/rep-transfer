#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=00-23:59
#SBATCH --mem=4000M
#SBATCH --account=def-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anffany@ualberta.ca
#SBATCH --job-name=test_train
#SBATCH --output=test_train.out

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif ./tests/test_train.sh

