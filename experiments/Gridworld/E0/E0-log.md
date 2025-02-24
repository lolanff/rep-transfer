#  Experiment E0 -- Playing around
##  Phase P0
### Objectives: 
- Familiarize with visualizing different learning curves. 
- Determine if agents with suboptimal learning rate (alpha) can still learn.
### Methods: 
- Compare learning curves of DQN-ReLU agents in two gridworlds with different goal locations, A & B.
- Compare learning curves of DQN-ReLU agents with the best alpha vs the second best alpha.
### Observations: 
- Based on the map, Goal B requires one more turn than Goal A to reach. 
- Agents' performance in Map B shows a larger variance (across 5 random seeds). 
### Conclusions & Outlooks: 
- No, agents with second best alpha don't seem to solve their tasks after 30,000 steps.
- Can they learn given more steps? Will their feature representation be worse or just as good?

## Phase P1
### Objectives: 
- Compare NN representations of successful DQN-ReLU agents trained at different learning rates
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
- For alpha=0.0001 (best), the feature maps are not instructive visually. 
- For alpha=0.0003 (second best), it can be seen that the feature maps highlight the approximate location of the agent. 
### Conclusions & Outlooks: 
- Agents with similar learning rates, if both successfully trained, learned similar feature representations.
- It appears that the feature maps serve to locate the agent. It makes sense. Since the walls and the goal never move, the important feature is the agent's location. As long as the agent knows where it is, it can decide an optimal action. 
- Why then does the feature representation only transfer well to similar tasks (in the ReLU case, according to Han et al)? One possible reason is that the feature representation may have encoded some other, map-specific information, e.g. "the bottom right region is good!". So the further the new goal is from goal A, the less generalizable the feature reps are.
- I am looking forward to see the feature maps in the FTA case.

## Phase P2
### Objectives: 
- Familiarize with the quality metrics for the representation in the linear layer.
### Methods: 
- Define the quality metrics (sparsity, etc) in the plotting scripts directly or in utils/functions.py
- Use the feature nets trained in Phase P1, compute the quality metrics of individual state-features.
- Visualize the metrics by histograms (rather than averaging to a number)
### Observations: 
##### Sparsity
- On average, the second-best-alpha agents have sparser feature representation.
- This agrees with the prior observation of the greater sparsity in the feature maps of second-best-alpha agent.
##### Orthogonality
- None of the best-alpha agent's features are close to being orthogonal with each other! Whereas the second-best-alpha agent has quite many orthogonal feature pairs.
- Agents with alpha=0.0003 and seed=0,2 have pretty orthogonal and sparse features.
##### Dynamics Awareness
- There are 246 out of 1044 transitions in which the agent walks into a wall and remains in the same state. I think we should exclude these transitions.
- I also assumed that the random state should be different from the current state.
- Dynamics awareness are consistently greater in the alpha=0.0003 agents.
- The agent with alpha=0.0003 and seed=0 has distinct states with the same features (as evidenced by "mean=inf" in the plots). I think these are the trivial features indicated in the orthogonality plot.
##### Complexity Reduction and Diversity
- These two quantities are essentially the same. One difference is that Diversity normalizes distances by the max distance, while Complexity reduction, somewhat arbitrarily, normalizes by  the largest Lipschitz ratio across all representations, in order to keep Complexity Reduction greater than 0. The second difference is that Diversity manually truncates any Lipschitz ratios greater than 1.
- I think that these two quantitites can merge into one: (1 - mean(Dv/Ds)), where Dv and Ds are distances normalized by their respective max distances. As seen in the plots, the distributions of normalized Lipschitz ratios often show characteristic peaks centered around 1. In contrast, un-normalized Lipschitz ratios have peaks at random places. 
- It makes sense to normalize distances in the representation space when comparing across different representations. The value network can trivially scale up or down the features, so the absolute values of the distances do not matter. 
### Conclusions & Outlooks: 
- Hyperparameter sweeping optimizes for faster learning, but not necessarily for better feature representation!
- Complexity Reduction and Diversity are very similar in their formulation, which may have led to the similarity of their respective plots in Fig. 5 (of Han et al's paper). I think that they can be merged into one metric. 
- "Non-interference" must be computed during training, and Adam said it doesn't quite make sense, so it will be ignored for now.

