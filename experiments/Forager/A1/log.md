This is a feasibility study hyper-sweep across forager two biomes large. The network architecture are not the same as the one used in the paper


Summary:

The main challenge is the credit assignment. From the videos of experiments, one phenomnenon is that the agent keep repeats a loop of actions while avoiding rewards even if that is within field of view, then at one point continue onward, possibly thank to epsilon-greedy.

Some ideas:

Only train with last step -> avoid update the network with only << T truncated sequence
Reduce buffer size -> make buffer more up to date
Use burn in -> avoid update the network with only < T truncated sequence


Here are the comparative experiements:

== Doubled sequence length compared to P1, from 128 to 256.
<= Duadrupled batch size compared to P1, from 2 to 8.
>= Reduced update frequency compared to P1, from 4 to 1.
== Changed activation compared to P1, from ReLU to FTA.
>= Dramatically reduced target update freqeuncy compared to P1, from 128 to 1.
