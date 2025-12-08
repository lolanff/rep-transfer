import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'results.csv')

# Read the CSV file
df = pd.read_csv(csv_path, index_col=0)

# Remove the last part after the last hyphen from algorithm names
df['algorithm'] = df['algorithm'].apply(lambda x: '-'.join(x.rsplit('-', 1)[:-1]))

# Calculate mean and 95% bootstrap confidence interval for each algorithm
from scipy import stats

def bootstrap_ci(data):
    """Calculate bootstrap confidence interval using scipy"""
    res = stats.bootstrap(
        (data,),
        np.mean,
        n_resamples=10000,
        confidence_level=0.95,
        method='percentile'
    )
    # Return the error bar size (difference from mean to upper bound)
    return res.confidence_interval.high - np.mean(data)

grouped = df.groupby('algorithm')['State Awareness']
means = grouped.mean()
stds = grouped.std()
sems = grouped.sem()
n_samples = grouped.count()
# Calculate bootstrap 95% CI
ci_95 = grouped.apply(lambda x: bootstrap_ci(x.values))

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars with 95% CI error bars
x_pos = np.arange(len(means))
bars = ax.bar(x_pos, means, yerr=ci_95, capsize=5, alpha=0.7, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Customize the plot
ax.set_xlabel('Algorithm', fontsize=16, fontweight='bold')
ax.set_ylabel('State Awareness', fontsize=16, fontweight='bold')
ax.set_title('State Awareness by Algorithm', fontsize=18, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for i, (mean, ci) in enumerate(zip(means, ci_95)):
    ax.text(i, mean + ci + 0.02, f'{mean:.3f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save the plot
output_path = os.path.join(script_dir, 'state_awareness_barplot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also display the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("=" * 60)
for alg in means.index:
    print(f"{alg}:")
    print(f"  Mean:   {means[alg]:.4f}")
    print(f"  Std:    {stds[alg]:.4f}")
    print(f"  SEM:    {sems[alg]:.4f}")
    print(f"  95% CI: {ci_95[alg]:.4f}")
    print(f"  n:      {n_samples[alg]}")
    print()
