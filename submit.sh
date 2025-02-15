#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=00-23:59
#SBATCH --mem=8G
#SBATCH --account=def-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anffany@ualberta.ca
#SBATCH --job-name=best-ScratchRelu
#SBATCH --output=best-ScratchRelu.out

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif ./scripts/best-ScratchRelu.sh