import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from experiment.tools import parseCmdLineArgs
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


from PyExpPlotting.matplot import save, setDefaultConference
import rlevaluation.hypers as Hypers
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return

setDefaultConference('jmlr')

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()
    
    df = pd.read_csv(f'{path}/combined_collector.csv', index_col=0)
    df['Experiment'] = df['Environment'] + df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True)

    fig, ax = plt.subplots(1, 1)
    plt.title("Performance vs Task Similarity")
    plt.xlabel("Task Similarity")
    plt.ylabel("Performance (Score)")
    plt.grid(True)

    relu_color               = "#66c2a5"  # teal
    fta_color                = "#fc8d62"  # orange
    relu_scratch_color       = "#8da0cb"  # lavender blue
    fta_scratch_color        = "#e78ac3"  # pink
    relu_dqn_color           = "#a6d854"  # green
    fta_dqn_color            = "#ffd92f"  # yellow
    relu_dqn_scratch_color   = "#e5c494"  # tan
    fta_dqn_scratch_color    = "#b3b3b3"  # gray

    alpha = 0.1  # Smoothing factor

    def compute_bootstrap_ci(data):
        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=data[:, None],
            statistic=Statistic.mean,
        )
        return res.ci[0], res.ci[1]

    # Compute mean and CI before smoothing
    grouped = df.groupby(['Experiment', 'Goal'])
    mean_ci_data = []

    for (experiment, goal), group in grouped:
        auc_values = group.groupby('Run')['AUC'].mean()
        mean_auc = auc_values.mean()
        lower_ci, upper_ci = compute_bootstrap_ci(auc_values.values)
        mean_ci_data.append((experiment, goal, mean_auc, lower_ci, upper_ci))

    # Convert to DataFrame and apply smoothing
    ci_df = pd.DataFrame(mean_ci_data, columns=['Experiment', 'Goal', 'Mean_AUC', 'Lower_CI', 'Upper_CI'])
    ci_df['Smoothed_AUC'] = ci_df.groupby('Experiment')['Mean_AUC'].transform(lambda x: x.ewm(alpha=alpha).mean())
    ci_df['Smoothed_Lower_CI'] = ci_df.groupby('Experiment')['Lower_CI'].transform(lambda x: x.ewm(alpha=alpha).mean())
    ci_df['Smoothed_Upper_CI'] = ci_df.groupby('Experiment')['Upper_CI'].transform(lambda x: x.ewm(alpha=alpha).mean())

    # Plot results
    for experiment, group in ci_df.groupby('Experiment'):
        print(experiment)
        name = str(experiment).lower()
        is_drqn = 'drqn' in name
        is_fta = 'fta' in name
        is_scratch = 'scratch' in name

        color_map = {
            (True, True, True): fta_scratch_color,
            (True, True, False): fta_color,
            (True, False, True): relu_scratch_color,
            (True, False, False): relu_color,
            (False, True, True): fta_dqn_scratch_color,
            (False, True, False): fta_dqn_color,
            (False, False, True): relu_dqn_scratch_color,
            (False, False, False): relu_dqn_color,
        }

        color = color_map[(is_drqn, is_fta, is_scratch)]
        print(color)
        plt.plot(group['Goal'], group['Smoothed_AUC'], color=color, alpha=0.6)
        plt.fill_between(group['Goal'], group['Smoothed_Lower_CI'], group['Smoothed_Upper_CI'], color=color, alpha=0.2)

    plt.scatter([], [], color=relu_color, label="DRQN ReLU", alpha=0.8)
    plt.scatter([], [], color=fta_color, label="DRQN FTA", alpha=0.8)
    plt.scatter([], [], color=relu_scratch_color, label="DRQN ReLU (Scratch)", alpha=0.8)
    plt.scatter([], [], color=fta_scratch_color, label="DRQN FTA (Scratch)", alpha=0.8)
    plt.scatter([], [], color=relu_dqn_color, label="DQN ReLU", alpha=0.8)
    plt.scatter([], [], color=fta_dqn_color, label="DQN FTA", alpha=0.8)
    plt.scatter([], [], color=relu_dqn_scratch_color, label="DQN ReLU (Scratch)", alpha=0.8)
    plt.scatter([], [], color=fta_dqn_scratch_color, label="DQN FTA (Scratch)", alpha=0.8)

    # Add legend and save plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.ylim(4.5, 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(
        save_path=f'{path}/plots',
        plot_name='smoothed_transfer_auc',
        f=fig,
        height_ratio=2/3,
    )

    # Plot individual run lines
    fig2, ax2 = plt.subplots(1, 1)
    plt.title("Individual Run AUC vs Task Similarity")
    plt.xlabel("Task Similarity")
    plt.ylabel("AUC")
    plt.grid(True)
    for experiment, group in df.groupby('Experiment'):
        name = str(experiment).lower()
        is_drqn = 'drqn' in name
        is_fta = 'fta' in name
        is_scratch = 'scratch' in name

        color = color_map[(is_drqn, is_fta, is_scratch)]
        for run, run_group in group.groupby('Run'):
            run_auc = run_group.groupby('Goal')['AUC'].mean()
            plt.plot(run_auc.index, run_auc.values, color=color, alpha=0.3)

    # Add legend for experiments
    for experiment in df['Experiment'].unique():
        name = str(experiment).lower()
        is_drqn = 'drqn' in name
        is_fta = 'fta' in name
        is_scratch = 'scratch' in name
        color = color_map[(is_drqn, is_fta, is_scratch)]
        plt.scatter([], [], color=color, label=experiment, alpha=0.8)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.ylim(4.5, 10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    save(
        save_path=f'{path}/plots',
        plot_name='individual_transfer_auc',
        f=fig2,
        height_ratio=2/3,
    )
