Analyse the property of representation

1. Have a random agent to collect 100,000 steps of history, then subsample 1,000 sequences of length 100.
    -- Have a DRQN agent with epsilon = 1 to run 100,000, save the check point
    -- load the checkpoint and get the buffer

2. Run each model on these to collect the represenation of the final step. Since the network handles the episode boundaries -- by forgetting the past, no special modification is needed.