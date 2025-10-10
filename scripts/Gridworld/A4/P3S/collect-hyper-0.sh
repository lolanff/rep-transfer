#!/bin/bash
set -e

rsync -aP --exclude=results --exclude=.* . ${SLURM_TMPDIR}
mkdir -p ${SLURM_TMPDIR}/results/Gridworld/A4/P3S
rsync -aP results/Gridworld/A4/P3S/ ${SLURM_TMPDIR}/results/Gridworld/A4/P3S

CURDIR=$(pwd)
cd ${SLURM_TMPDIR}
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif python experiments/Gridworld/A4/P3S/collect_hyper.py
cd "$CURDIR"

rsync -aP ${SLURM_TMPDIR}/experiments/Gridworld/A4/P3S/hyperparameter_collector.csv experiments/Gridworld/A4/P3S/hyperparameter_collector.csv