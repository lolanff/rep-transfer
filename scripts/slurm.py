import sys
import os
sys.path.append(os.getcwd() + '/src')

import math
import time
import argparse
import dataclasses
import PyExpUtils.runner.Slurm as Slurm
import experiment.ExperimentModel as Experiment

from functools import partial
from PyExpUtils.utils.generator import group
from PyExpUtils.runner.utils import approximate_cost, gather_missing_indices

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, required=True)
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')
parser.add_argument('--debug', action='store_true', default=False)

cmdline = parser.parse_args()

# -------------------------------
# Load cluster configuration
# -------------------------------
with open(cmdline.cluster, 'r') as f:
    import json
    slurm_config = json.load(f)

# ----------------
# Scheduling logic
# ----------------
threads = slurm_config.get('threads_per_task', 1)

# compute how many "tasks" to clump into each job
groupSize = int(slurm_config['cores'] / threads) * slurm_config['sequential']

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm_config['time'].split(':')
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing
missing = gather_missing_indices(cmdline.e, cmdline.runs, loader=Experiment.load)

# compute cost
memory = int(slurm_config['mem_per_core'].replace('G', '')) * 1024
compute_cost = partial(approximate_cost, cores_per_job=slurm_config['cores'], mem_per_core=memory, hours=total_hours)
cost = sum(compute_cost(math.ceil(len(job_list) / groupSize)) for job_list in missing.values())
print(f"Expected to use {cost*365.25:.2f} core days.")

# Generate job scripts
cwd = os.getcwd()
project_name = os.path.basename(cwd)

def generate_job_script(exp_path, task_indices):
    """Generate a SLURM job script for a group of tasks"""
    return f"""#!/bin/bash
#SBATCH --account={slurm_config['account']}
#SBATCH --time={slurm_config['time']}
#SBATCH --cpus-per-task={slurm_config['threads_per_task']}
#SBATCH --mem-per-cpu={slurm_config['mem_per_core']}
#SBATCH --output=job_%A_%a.out
#SBATCH --signal=B:SIGTERM@180

module load apptainer

cd {cwd}

# Map SLURM_ARRAY_TASK_ID to actual task index
declare -a task_indices=(${' '.join(map(str, task_indices))})
task_idx=${{task_indices[$SLURM_ARRAY_TASK_ID]}}

apptainer exec -C -B .:${{HOME}} -W ${{SLURM_TMPDIR}} pyproject.sif python {cmdline.entry} -e {exp_path} --save_path {cmdline.results} -i $task_idx
"""

# Create scripts directory if it doesn't exist
os.makedirs('slurm_scripts', exist_ok=True)

# Generate scripts for each experiment
for i, (exp_path, indices) in enumerate(missing.items()):
    # Split indices into groups
    for j, group_indices in enumerate(group(indices, groupSize)):
        task_list = list(group_indices)
        script = generate_job_script(exp_path, task_list)
        
        filename = f"slurm_scripts/job_{i}_{j}.sh"
        with open(filename, 'w') as f:
            f.write(script)
        
        print(f"\nGenerated {filename} for experiment {exp_path}")
        print(f"To submit this job, run:")
        print(f"sbatch --array=0-{len(task_list)-1} {filename}")