#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import bootstrap
import numpy as np

cur_path = Path()

df = pd.read_csv(cur_path / 'collector.csv', index_col=0)
df['Experiment'] = df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True)
#%%
plt.figure(figsize=(12, 8))
plt.title("Performance vs Task Similarity")
plt.xlabel("Task Similarity")
plt.ylabel("Performance (Score)")
plt.grid(True)

relu_aux_color = 'blue'
relu_color = 'red'
relu_aux_no_early_saving_color = 'green'
relu_no_early_saving_color = 'yellow'

# Group by experiment and seed, plot lines for each group
grouped = df.groupby(['Environment', 'Experiment', 'Seed'])
for (environment, experiment, seed), group in grouped:
    group['Goal'] = pd.to_numeric(group['Goal'], errors='coerce')
    print(group['Goal'])
    group = group.sort_values(by='Goal', ascending=True).dropna(subset=['Goal', 'AUC'])
    
    color = (
        relu_aux_no_early_saving_color if 'vf5' in experiment and 'no_early' in environment 
        else relu_no_early_saving_color if 'no_early' in environment 
        else relu_aux_color if 'vf5' in experiment 
        else relu_color
    )
    
    plt.plot(group['Goal'], group['AUC'], color=color, marker='o', linewidth=5, alpha=1)

# Use plt.plot for legend (not scatter)
plt.plot([], [], color=relu_aux_color, label="ReLU-Aux", linewidth=5, alpha=0.8)
plt.plot([], [], color=relu_color, label="ReLU", linewidth=5, alpha=0.8)
plt.plot([], [], color=relu_aux_no_early_saving_color, label="ReLU-Aux-No-Early-Saving", linewidth=5, alpha=0.8)
plt.plot([], [], color=relu_no_early_saving_color, label="ReLU-No-Early-Saving", linewidth=5, alpha=0.8)

# Add legend and show the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(cur_path/"plot.png")
plt.show()
# %%
plt.figure(figsize=(12, 8))
plt.title("Performance vs Task Similarity")
plt.xlabel("Task Similarity")
plt.ylabel("Performance (Score)")
plt.grid(True)

relu_aux_color = 'blue'
relu_color = 'red'

alpha = 0.1

grouped = df.groupby(['Experiment', 'Seed'])
for (experiment, seed), group in grouped:
    group = group.sort_values(by='Goal')
    group['Smoothed_AUC'] = group['AUC'].ewm(alpha=alpha).mean()
    color = relu_aux_color if 'vf5' in experiment else relu_color
    plt.plot(group['Goal'], group['Smoothed_AUC'], color=color, alpha=0.6)

plt.scatter([], [], color=relu_aux_color, label="ReLU-Aux", alpha=0.8)
plt.scatter([], [], color=relu_color, label="ReLU", alpha=0.8)

# Add legend and save plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(cur_path / "plot_smoothed.png")
plt.show()


# %%
plt.figure(figsize=(12, 8))
plt.title("Performance vs Task Similarity")
plt.xlabel("Task Similarity")
plt.ylabel("Performance (Score)")
plt.grid(True)

relu_aux_color = 'blue'
relu_color = 'red'

alpha = 0.1  # Smoothing factor
ci_percentile = 0.95

def compute_bootstrap_ci(data, ci=0.95):
    if len(data) > 1:
        ci_bounds = bootstrap((data,), np.mean, confidence_level=ci, n_resamples=1000, method='percentile').confidence_interval
        return ci_bounds.low, ci_bounds.high
    else:
        return data[0], data[0]  # Return the same value if only one data point exists

# Compute mean and CI before smoothing
grouped = df.groupby(['Experiment', 'Goal'])
mean_ci_data = []

for (experiment, goal), group in grouped:
    auc_values = group.groupby('Seed')['AUC'].mean()
    mean_auc = auc_values.mean()
    lower_ci, upper_ci = compute_bootstrap_ci(auc_values.values, ci_percentile)
    mean_ci_data.append((experiment, goal, mean_auc, lower_ci, upper_ci))

# Convert to DataFrame and apply smoothing
ci_df = pd.DataFrame(mean_ci_data, columns=['Experiment', 'Goal', 'Mean_AUC', 'Lower_CI', 'Upper_CI'])
ci_df['Smoothed_AUC'] = ci_df.groupby('Experiment')['Mean_AUC'].transform(lambda x: x.ewm(alpha=alpha).mean())
ci_df['Smoothed_Lower_CI'] = ci_df.groupby('Experiment')['Lower_CI'].transform(lambda x: x.ewm(alpha=alpha).mean())
ci_df['Smoothed_Upper_CI'] = ci_df.groupby('Experiment')['Upper_CI'].transform(lambda x: x.ewm(alpha=alpha).mean())

# Plot results
for experiment, group in ci_df.groupby('Experiment'):
    color = relu_aux_color if 'vf5' in experiment else relu_color
    plt.plot(group['Goal'], group['Smoothed_AUC'], color=color, alpha=0.6)
    plt.fill_between(group['Goal'], group['Smoothed_Lower_CI'], group['Smoothed_Upper_CI'], color=color, alpha=0.2)

plt.scatter([], [], color=relu_aux_color, label="ReLU-Aux", alpha=0.8)
plt.scatter([], [], color=relu_color, label="ReLU", alpha=0.8)

# Add legend and save plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(cur_path / "plot_avg_smoothed.png")
plt.show()

# %%
