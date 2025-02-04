#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --time=00-23:59
#SBATCH --mem=8G
#SBATCH --account=def-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anffany@ualberta.ca
#SBATCH --job-name=E1-P0
#SBATCH --output=E1-P0.out

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif ./scripts/E1-P0.sh