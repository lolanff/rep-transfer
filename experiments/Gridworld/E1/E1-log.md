# Experiment E1 -- Reproducing Fig. 3 in Han et al.'s paper
## Phase P0
### Objectives: 
- Reproduce Scratch(Relu) and Scratch(FTA)
### Methods: 
- Found Han's dataset that sorts goals by task similarity. Assigned coordinates to goals.
- Scratch curves do not use auxiliary tasks.
- Transfer tasks run for 100,000 steps, including baselines.
- Assume that eta = 0.2 for Scratch(FTA)
- Hypersweeps over alpha = [0.001, 0.0003, 0.0001, 0.00003, 0.00001]
- Hyperparameters are selected by AUC as defined in the paper