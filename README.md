Transferable Neural Network Representations in Reinforcement Learning
=====================
To schedule a big experiment on Cedar: 
1. Make sure your project directory contains apptainer image "pyproject.sif"
2. Update "submit.sh" for your experiment (replace the json file)
3. Run "sbatch submit.sh"

submit.sh calls slurm.py, which submits multiple jobs, each containing a portion of all tasks.

---
Acknowledgement: This repository is adapted from erfanMhi/LTA-Representation-Properties, andnp/rl-control-template, and steventango/sparse-feature-transfer.