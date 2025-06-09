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
    
    scratch_relu = [8.297999999999998, 8.832, 8.8, 8.873999999999999, 8.934000000000001, 8.728, 9.052000000000001, 8.352, 9.072, 8.628, 9.36, 9.15, 9.030000000000001, 9.326000000000002, 9.992, 9.062000000000001, 9.466, 8.522, 9.708000000000002, 9.012, 8.85, 8.758, 8.374, 8.806000000000001, 7.8, 8.412, 9.372, 7.413999999999999, 9.006, 9.187999999999999, 8.873999999999999, 8.606, 9.367999999999999, 9.19, 8.744]

    scratch_fta = [7.94, 8.572, 9.136000000000001, 8.328, 8.568, 8.786000000000001, 8.706, 7.395999999999999, 4.96, 8.852, 9.186, 9.02, 8.534, 9.23, 8.652000000000001, 8.623999999999999, 9.132000000000001, 8.202000000000002, 9.156, 7.38, 8.320000000000002, 8.138, 8.882, 8.192, 8.138, 8.309999999999999, 8.84, 7.008, 9.248000000000001, 8.922, 8.562000000000001, 8.373999999999999, 8.998, 8.314, 8.402]

    alpha = 0.1
    smooth_relu = pd.Series(scratch_relu).ewm(alpha=alpha).mean()
    smooth_fta = pd.Series(scratch_fta).ewm(alpha=alpha).mean()

    fig, ax = plt.subplots(1, 1)
    plt.title("Performance vs Task Similarity")
    plt.xlabel("Task Similarity")
    plt.ylabel("Performance (Score)")
    plt.grid(True)

    relu_aux_color = 'blue'
    relu_color = 'green'
    relu_ff_scratch_color = 'brown'
    fta_color = 'yellow'
    fta_aux_color = 'red'
    fta_ff_scratch_color = 'purple'

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
    plt.scatter([], [], color=relu_ff_scratch_color, label="ReLU FF (Scratch)", alpha=0.8)
    plt.scatter([], [], color=fta_ff_scratch_color, label="FTA FF (Scratch)", alpha=0.8)

    subsampled_goals = list(range(0, 170 + 1, 5))

    plt.plot(subsampled_goals, smooth_relu, color=relu_ff_scratch_color, linestyle='-', alpha=0.8)
    plt.plot(subsampled_goals, smooth_fta, color=fta_ff_scratch_color, linestyle='-', alpha=0.8)

    # Add legend and save plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.ylim(4.5, 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save(
        save_path=f'{path}/plots',
        plot_name='smoothed_transfer_auc_with_dqn_scratch',
        f=fig,
        height_ratio=2/3,
    )
