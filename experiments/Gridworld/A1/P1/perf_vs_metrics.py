
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyExpPlotting.matplot import save, setDefaultConference
from experiment.tools import parseCmdLineArgs
import jax.numpy as jnp
import numpy as np
import time
import pandas as pd

setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

# Load the data files
collector_df = pd.read_csv(f'{path}/collector.csv', index_col=0)  # Update this with your file path
results_df = pd.read_csv(f'{path}/results.csv', index_col=0)  # Update this with your file path

# Step 1: Preprocess collector.csv
# Extract the base algorithm name by removing the last part (goal)
collector_df['Base Algorithm'] = collector_df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True)

# Average AUC by base name and Seed
avg_scores = collector_df.groupby(['Base Algorithm', 'Seed'])['AUC'].mean().reset_index()
avg_scores.rename(columns={'Base Algorithm': 'algorithm', 'Seed': 'seed', 'AUC': 'Average Score'}, inplace=True)

# Step 2: Map the algorithm names for consistency
# Create a mapping dictionary
name_mapping = {
    'DQN-ReLU-transfer': 'DQN-ReLU-A',
    'DQN-ReLU-vf5-transfer': 'DQNAux-ReLU-A'
}

# Apply the mapping to the algorithm column
avg_scores['algorithm'] = avg_scores['algorithm'].map(name_mapping)

# Step 3: Merge averaged scores with metrics from results.csv
merged_df = pd.merge(avg_scores, results_df, on=['algorithm', 'seed'])

# Step 4: Combine Aux and No Aux without separation
combined_data = merged_df

# Step 5: Plot the results without separating Aux and No Aux
metrics = ['Complexity Reduction', 'Dynamics Awareness', 'Diversity', 'Orthogonality', 'Sparsity']
fig, axs = plt.subplots(2, len(metrics), figsize=(20, 10), sharex='col', sharey='row', gridspec_kw={'height_ratios': [4, 1]})
fig.suptitle('AUC vs Representation Properties', fontsize=24, fontweight='bold')

# Plot each metric
for col, metric in enumerate(metrics):
    # Scatter Plot for Combined Data
    ax = axs[0, col]
    ax.scatter(combined_data[metric], combined_data['Average Score'], 
               color=combined_data['algorithm'].map({'DQN-ReLU-A': 'blue', 'DQNAux-ReLU-A': 'blue'}), 
               alpha=0.6)
    
    # Mark overall top 3 points regardless of Aux status
    top3 = combined_data.nlargest(3, 'Average Score')
    ax.scatter(top3[metric], top3['Average Score'], color='green', edgecolor='black', marker='*', s=150, label='Top 3')
    
    # Set titles for the top row
    ax.set_title(metric, fontsize=18)
    
    # Histogram on the bottom row
    ax_hist = axs[1, col]
    ax_hist.hist(combined_data[metric], bins=20, color='gray', alpha=0.6)
    
    # Set y-axis labels for the first column
    if col == 0:
        axs[0, col].set_ylabel('AUC', fontsize=18)
        axs[1, col].set_ylabel('Density', fontsize=18)

# Add legend only once
axs[0, 0].legend(fontsize=14)

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{path}/perf_vs_metrics.png')
