#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=0-23:59
#SBATCH --mem=64G
#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinze5@ualberta.ca
#SBATCH --job-name=A2-P0
#SBATCH --output=A2-P0-%j.out

export OMP_NUM_THREADS=1

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif ./scripts/A2-P0.sh

