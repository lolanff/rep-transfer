#!/bin/bash

#SBATCH --account=aip-amw8
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=8
#SBATCH --time=01:00:00
#SBATCH --export=path="/home/xiongxz/project/rep-transfer"

cp $path/pyproject.toml $SLURM_TMPDIR/

cd $SLURM_TMPDIR
module load python/3.11 arrow/19 gcc opencv rust/1.70.0 swig
python -m venv .venv
source .venv/bin/activate
module load rust swig arrow/19 gcc opencv
.venv/bin/pip install -e .

cp -r .venv $path/

pip freeze
