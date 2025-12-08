#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=0-23:59
#SBATCH --mem=64G
#SBATCH --account=aip-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinze5@ualberta.ca
#SBATCH --job-name=scripts/Gridworld/A4/P7/collect.sh
#SBATCH --output=scripts/Gridworld/A4/P7/collect.sh-%a-%A.out
#SBATCH --array=0-0

export OMP_NUM_THREADS=1
# ${SLURM_ARRAY_TASK_ID}

source .venv/bin/activate
./scripts/Gridworld/A4/P7/collect-${SLURM_ARRAY_TASK_ID}.sh
