train the RNN rep and value function on the source task
freeze the rep
for i =1:n
train the a new value function on transfer task n
at the end of training, use the current policy to collect all these states
compute state awareness and the AUC in task n
average metric and AUC over n