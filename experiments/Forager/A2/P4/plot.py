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
    # Extract FOV and algorithm type
    df['FOV'] = df['Algorithm'].str.split('-').str[1].astype(int)
    df['AlgoType'] = df['Algorithm'].str.split('-').str[0]

    dqn_color               = "#66c2a5"  # teal
    drqn_color              = "#fc8d62"  # orange
    
    def compute_bootstrap_ci(data):
        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=data[:, None],
            statistic=Statistic.mean,
        )
        return res.ci[0][0], res.ci[1][0]

    # Compute mean AUC and CI by algorithm type and FOV
    mean_ci_data = []
    for (algo, fov), grp in df.groupby(['AlgoType', 'FOV']):
        auc_vals = grp['P_AUC'].values
        mean_auc = auc_vals.mean()
        lower_ci, upper_ci = compute_bootstrap_ci(auc_vals)
        mean_ci_data.append((algo, fov, mean_auc, lower_ci, upper_ci))
    ci_df = pd.DataFrame(mean_ci_data, columns=['AlgoType', 'FOV', 'Mean_AUC', 'Lower_CI', 'Upper_CI'])

    print(ci_df)
    # Plot performance vs FOV
    fig, ax = plt.subplots()
    plt.title("Performance vs FOV")
    plt.xlabel("FOV")
    plt.ylabel("AUC")
    for algo, color in [('DQN', dqn_color), ('DRQN', drqn_color)]:
        data = ci_df[ci_df['AlgoType'] == algo].sort_values('FOV')
        ax.plot(data['FOV'], data['Mean_AUC'], label=algo, color=color)
        ax.fill_between(data['FOV'], data['Lower_CI'], data['Upper_CI'], color=color, alpha=0.2)
        # Add horizontal reference lines with legend entries
    ax.axhline(1.4, linestyle='--', color='grey', linewidth=1, label='Search Oracle (roughly)')
    ax.axhline(0.8, linestyle='--', color='lightgrey', linewidth=1, label='Search Nearest (roughly)')
    plt.legend()

    ax.set_ylim(0, 1.5)
    ax.set_xticks(list(range(3, 16, 2)))

    save(
        save_path=f'{path}/plots',
        plot_name='performance_vs_fov',
        f=fig,
        height_ratio=2/3,
    )
