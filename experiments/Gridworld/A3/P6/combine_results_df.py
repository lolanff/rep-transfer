#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import bootstrap
import numpy as np
import pandas as pd

cur_path = Path()

fta_df = pd.read_csv(cur_path / 'results_ff.csv', index_col=0)
relu_df = pd.read_csv(cur_path / 'results_rnn.csv', index_col=0)

df = pd.concat([relu_df, fta_df], ignore_index=True)
df['Complexity Reduction'] = 1 - (df['Raw Complexity Reduction'] / df['Raw Complexity Reduction'].max())

df.to_csv('combined_results.csv')


# %%
