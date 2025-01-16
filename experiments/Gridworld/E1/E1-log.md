## Experiment E1, Phase P0
### Objectives: 
- Train two DQN agents, with ReLU(L) and FTA activations respectively
- Analyze their feature nets (conv kernels, conv feature maps, quality metrics for the representation)
### Methods: 
- Sweep over hyperparameter alpha. Save all checkpoints.
- For DQN-FTA, fix hidden = 32 and eta = 0.2. For DQN-ReLU(L), fix hidden = 640.
