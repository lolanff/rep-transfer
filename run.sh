#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=0-23:59
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinze5@ualberta.ca
#SBATCH --job-name=A1-P0-DQNAux
#SBATCH --output=A1-P0-DQNAux.out

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif ./scripts/A1-P0-DQNAux.sh

