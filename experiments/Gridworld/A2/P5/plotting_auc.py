import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

import os
import sys
sys.path.append(os.getcwd() + '/src')

from experiment.tools import parseCmdLineArgs

from PyExpPlotting.matplot import save, setDefaultConference
setDefaultConference('neurips')

# Increase default font sizes for readability
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 12
})

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    df = pd.read_csv(f'{path}/collector.csv', index_col=0)

    def classify_group(algo):
        algo = algo.lower()
        if 'drqn' in algo and 'fta' in algo:
            return 'DRQN FTA'
        elif 'drqn' in algo:
            return 'DRQN ReLU'
        elif 'fta' in algo:
            return 'DQN FTA'
        else:
            return 'DQN ReLU'

    df['Group'] = df['Algorithm'].apply(classify_group)

    def extract_nums(algo):
        nums = re.findall(r'(\d+)', algo)
        return (int(nums[-2]), int(nums[-1])) if len(nums) >= 2 else (32, 1)

    df['Batch'], df['Seq'] = zip(*df['Algorithm'].apply(extract_nums))
    df = df.dropna(subset=['Batch', 'Seq']).reset_index(drop=True)
    df['Batch'] = df['Batch'].astype(int)
    df['Seq'] = df['Seq'].astype(int)

    df['Label'] = df.apply(lambda r: f"BS={r['Batch']}, SL={r['Seq']}", axis=1)

    df = df.sort_values(['Group', 'Batch', 'Seq']).reset_index(drop=True)

    cmap = plt.get_cmap('tab20')

    groups = [['DRQN FTA', 'DQN FTA'], ['DRQN ReLU', 'DQN ReLU']]
    subsets = [(df[df['Group'].isin(groups[0])], groups[0], 'FTA'), (df[df['Group'].isin(groups[1])], groups[1], 'ReLU')]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(len(groups[0]) * 1.5, 6),
        sharey=True,
        sharex=True
    )

    for ax, (data_subset, envs, title) in zip(axes, subsets):
        # Plot DRQN variants
        drqn_envs = envs[:-1]
        labels_sub = data_subset[data_subset['Group'].isin(drqn_envs)]['Label'].unique()
        n_labels_sub = len(labels_sub)
        width_sub = 0.8 / n_labels_sub
        colors = [cmap(i) for i in range(n_labels_sub)]
        for i, lab in enumerate(labels_sub):
            sub = data_subset[data_subset['Label'] == lab]
            means, err_low, err_high = [], [], []
            for e in drqn_envs:
                row = sub[sub['Group'] == e]
                if not row.empty:
                    m = row['AUC'].values[0]
                    lo = row['AUC_low'].values[0]
                    hi = row['AUC_high'].values[0]
                else:
                    m, lo, hi = np.nan, np.nan, np.nan
                means.append(m)
                err_low.append(m - lo if not np.isnan(m) else 0)
                err_high.append(hi - m if not np.isnan(m) else 0)
            positions = np.arange(len(drqn_envs)) + (i - (n_labels_sub - 1) / 2) * width_sub
            ax.bar(positions, means, width_sub, color=colors[i], edgecolor='white',
                   yerr=[err_low, err_high], capsize=5, label=lab)
        # Plot the DQN baseline
        baseline_env = envs[-1]
        row = data_subset[data_subset['Group'] == baseline_env]
        if not row.empty:
            m = row['AUC'].values[0]
            lo = row['AUC_low'].values[0]
            hi = row['AUC_high'].values[0]
            base_idx = len(drqn_envs)
            ax.bar(base_idx, m, width_sub, color=colors[0], edgecolor='white',
                   yerr=[[m - lo], [hi - m]], capsize=5)

        ax.set_xticks(np.arange(len(envs)))
        ax.set_xticklabels([env.replace(' FTA', '').replace(' ReLU', '') + '\n' + ('BS=2\nSL=16' if 'drqn' in env.lower() else 'BS=32\nSL=1') for env in envs])
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[0].set_ylabel('AUC')

    save(
        save_path=f'{path}/plots',
        plot_name='auc_results',
        f=fig,
        width=0.7,
        height_ratio=2/3,
    )