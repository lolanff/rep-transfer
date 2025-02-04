Transferable Neural Network Representations in Reinforcement Learning
=====================
To schedule a big experiment on cedar, use slurm.py: 
1. Make sure your project directory contains apptainer image "pyproject.sif"
2. Set slurm requirements in clusters/cedar.json 
3. Run the following in a login node
* module load apptainer 
* apptainer exec -C -B .:$HOME pyproject.sif python scripts/slurm.py --cluster clusters/cedar.json --runs 5 -e experiments/Gridworld/E1/P0/DQN-Relu.json 
(replace the json file)
4. The slurm jobs, each containing a portion of all tasks, will be saved in slurm_scripts. To submit them all, run "./slurm_scripts/submit_all.sh"

---
Acknowledgement: This repository is adapted from erfanMhi/LTA-Representation-Properties, andnp/rl-control-template, and steventango/sparse-feature-transfer.