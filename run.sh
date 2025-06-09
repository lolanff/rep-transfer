#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --time=0-2:59
#SBATCH --mem=24G
#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinze5@ualberta.ca
#SBATCH --job-name=scripts/A3/P4S/scratch.sh
#SBATCH --output=scripts/A3/P4S/scratch.sh-%a-%A.out
#SBATCH --array=0-69%36

export OMP_NUM_THREADS=1
# ${SLURM_ARRAY_TASK_ID}

module load apptainer
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif ./scripts/A3/P4S/scratch-${SLURM_ARRAY_TASK_ID}.sh
