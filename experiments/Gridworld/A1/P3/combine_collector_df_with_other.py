#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import bootstrap
import numpy as np
import pandas as pd

cur_path = Path()

fta_df = pd.read_csv(cur_path / 'collector.csv', index_col=0)
relu_df = pd.read_csv(cur_path / '../P1/collector.csv', index_col=0)
fta_df['Experiment'] = fta_df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True)
relu_df['Experiment'] = relu_df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True)
relu_df['Goal'] = relu_df['Goal'] + 1
# %%
df = pd.concat([relu_df, fta_df], ignore_index=True)
df.to_csv('combined_collector.csv')
# %%
df
# %%
