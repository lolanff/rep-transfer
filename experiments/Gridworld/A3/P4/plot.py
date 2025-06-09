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
    
    df = pd.read_csv(f'{path}/collector.csv', index_col=0)
    df['Experiment'] = df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True)

    fig, ax = plt.subplots(1, 1)
    plt.title("Performance vs Task Similarity")
    plt.xlabel("Task Similarity")
    plt.ylabel("Performance (Score)")
    plt.grid(True)

    relu_aux_color = 'blue'
    relu_color = 'green'
    fta_color = 'yellow'
    fta_aux_color = 'red'

    alpha = 0.1  # Smoothing factor
    ci_percentile = 0.95

    def compute_bootstrap_ci(data, ci=0.95):
        if len(data) > 1:
            ci_bounds = bootstrap((data,), np.mean, confidence_level=ci, n_resamples=10000, method='percentile').confidence_interval
            return ci_bounds.low, ci_bounds.high
        else:
            return data[0], data[0]  # Return the same value if only one data point exists

    # Compute mean and CI before smoothing
    grouped = df.groupby(['Experiment', 'Goal'])
    mean_ci_data = []

    for (experiment, goal), group in grouped:
        auc_values = group.groupby('Run')['AUC'].mean()
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
        print(experiment)
        if 'fta' in str(experiment).lower():
            if 'aux' in str(experiment).lower():
                color = fta_aux_color
            else:
                color = fta_color
        else:
            if 'aux' in str(experiment).lower():
                color = relu_aux_color
            else:
                color = relu_color
        print(color)
        plt.plot(group['Goal'], group['Smoothed_AUC'], color=color, alpha=0.6)
        plt.fill_between(group['Goal'], group['Smoothed_Lower_CI'], group['Smoothed_Upper_CI'], color=color, alpha=0.2)

    plt.scatter([], [], color=relu_aux_color, label="ReLU-Aux", alpha=0.8)
    plt.scatter([], [], color=relu_color, label="ReLU", alpha=0.8)
    plt.scatter([], [], color=fta_aux_color, label="FTA-Aux", alpha=0.8)
    plt.scatter([], [], color=fta_color, label="FTA", alpha=0.8)

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
