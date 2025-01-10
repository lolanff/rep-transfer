##  Experiment E0, Phase P0
### Objectives: 
- Familiarize with visualizing different learning curves. 
- Determine if agents with suboptimal learning rate (alpha) can still learn.
### Methods: 
- Compare learning curves of DQN agents in two gridworlds with different goal locations, A & B.
- Compare learning curves of DQN agents with the best alpha vs the second best alpha.
### Observations: 
- Based on the map, Goal B requires one more turn than Goal A to reach. 
- Agents' performance in Map B shows a larger variance (across 5 random seeds). 
### Conclusions: 
- No, agents with second best alpha don't seem to solve their tasks after 30,000 steps.
- Can they learn given more steps? Will their feature representation be worse or just as good?

## Experiment E0, Phase P1
### Objectives: 
- Compare NN representations of successful agents trained at different learning rates
### Methods: 
- Train agents at learning rates 0.0001 (best from sweeping) and 0.0003 (second best)
- Use Goal location A
- Set a long training period: 100,000 steps
- Save checkpoints



