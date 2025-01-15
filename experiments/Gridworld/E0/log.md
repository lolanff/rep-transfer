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
### Conclusions & Outlooks: 
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
### Observations:
- Both agents learned. The best-alpha agent learned faster.
- The kernels in the first convolutional layer (which has three input channels) appear to have some structures (i.e. not completely random): straight lines, corners, isolated dots, etc.
- The kernels in both alpha-cases look similar, but the best-alpha-case shows higher contrast.
- The fact that the kernels are similar at different learning rates suggests that both agents ended up near the same minimum of the loss function. This makes sense because both agents start at the same initial position in the loss function landscape, and their learning rates are similar.
- For alpha=0.0001, the feature maps are not instructive visually. 
- For alpha=0.0003, it can be seen that the feature maps highlight the approximate location of the agent. 
### Conclusions & Outlooks: 
- Agents with similar learning rates, if both successfully trained, learned similar feature representations.
- It appears that the feature maps serve to locate the agent. It makes sense. Since the walls and the goal never move, the important feature is the agent's location. As long as the agent knows where it is, it can decide an optimal action. 
- Why then does the feature representation only transfer well to similar tasks? The feature representation must have encoded some other, map-specific information, e.g. "the bottom right region is good!". So the further the new goal is from goal A, the less generalizable the feature reps are.
- I am looking forward to see the feature maps in the fta case.