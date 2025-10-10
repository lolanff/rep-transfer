#!/bin/bash
set -e

rsync -aP --exclude=results --exclude=.* . ${SLURM_TMPDIR}
mkdir -p ${SLURM_TMPDIR}/results/Forager/A2/P3
rsync -aP results/Forager/A2/P3/ ${SLURM_TMPDIR}/results/Forager/A2/P3

CURDIR=$(pwd)
cd ${SLURM_TMPDIR}
apptainer exec -C -B .:${HOME} -W ${SLURM_TMPDIR} pyproject.sif python experiments/Forager/A2/P3/collect_hyper.py
cd "$CURDIR"

rsync -aP ${SLURM_TMPDIR}/experiments/Forager/A2/P3/hyperparameter_collector.csv experiments/Forager/A2/P3/hyperparameter_collector.csv