Transferable Neural Network Representations in Reinforcement Learning
=====================
To schedule a big experiment on Cedar: 
1. Go to your project directory, and make sure that it contains the apptainer image pyproject.sif
2. Add an executable shell script to ./scripts that calls slurm.py. For example, scripts/E1-P0.sh contains
python scripts/slurm.py --cluster clusters/cedar.json --runs 5 -e experiments/Gridworld/E1/P0/DQN-Relu.json 
(replace the json file with yours)
3. Run the following directly on login terminal
* module load apptainer
* apptainer exec -C -B .:$HOME -W $SLURM_TMPDIR pyproject.sif scripts/E1-P0.sh


---
Acknowledgement: This repository is adapted from erfanMhi/LTA-Representation-Properties, andnp/rl-control-template, and steventango/sparse-feature-transfer.