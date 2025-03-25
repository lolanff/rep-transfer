test replay buffer

Use GRU(RTU) layer instead of LSTM,  RTU > GRU > LSTM
Replay buffer: 
take random index then sample sequences after that
reset hidden state once reach termianl so it's ok to have terminal state in the middle of sequence

Test on Memory, then FOT of Han's, then forager domain, it's possible to not able to solve initially, then comparison between solvable and insolvable representation
The core is what representation is good for continual learning
can it do continual learning task? what's the propersties

choose longer than the length of the optimal policy, choose based on reasoning, then empirically test it, using 2^n, maybe just 2, it's not a param to sweep. if recreating, just use the same as the paper

what's the property in FOT vs POT for the representation of the hidden state.