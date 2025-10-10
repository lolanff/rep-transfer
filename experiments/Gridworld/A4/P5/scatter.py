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
    
    score_raw_df = pd.read_csv(f'{path}/combined_collector.csv', index_col=0)
    score_raw_df = score_raw_df[score_raw_df['Environment'].str.contains("transfer")]
    score_raw_df['Experiment'] = score_raw_df['Algorithm'].str.replace(r"-(-|[0-9])+", "", regex=True) + 'Partial'
    
    grouped = score_raw_df.groupby(['Experiment', 'Run'])
    mean_data = []

    for (experiment, seed), group in grouped:
        mean_auc = group['AUC'].mean()
        mean_data.append((experiment, seed, mean_auc))
        
    score_df = pd.DataFrame(mean_data, columns=['algorithm', 'seed', 'AUC'])
    
    rep_df = pd.read_csv(f'{path}/results.csv', index_col=0)

    df = rep_df.merge(score_df, on=['algorithm', 'seed'])
    
    features = [
        'Dynamics Awareness',
        'Diversity',
        'Orthogonality',
        'Sparsity',
        'Complexity Reduction'
    ]

    dqn_fta  = df[df['algorithm'].str.contains('DQN-FTA')]
    drqn_fta  = df[df['algorithm'].str.contains('DRQN-FTA')]
    dqn_relu = df[df['algorithm'].str.contains('DQN-ReLU')]
    drqn_relu = df[df['algorithm'].str.contains('DRQN-ReLU')]
    
    df_dict = {
        'FTA': {
            'DQN': dqn_fta,
            'DRQN': drqn_fta
        },
        'ReLU': {
            'DQN': dqn_relu,
            'DRQN': drqn_relu
        }
    }

    for activation, info_dict in df_dict.items():
        fig, axes = plt.subplots(1, len(features), sharey=True)
        for ax, feat in zip(axes, features):
            ax.scatter(info_dict['DQN'][feat],  info_dict['DQN']['AUC'],  label=f'DQN-{activation}',  marker='o')
            ax.scatter(info_dict['DRQN'][feat],  info_dict['DRQN']['AUC'],  label=f'DRQN-{activation}', marker='X')
            ax.set_xlabel(feat)
        axes[0].set_ylabel('AUC')
        axes[-1].legend(loc='upper left')
        fig.suptitle(f'AUC vs Representation Properties ({activation})', y=1.05)
        
        save(
            save_path=f'{path}/plots',
            plot_name=f'auc_vs_rep_prop_{activation}',
            f=fig,
            width=2,
            height_ratio=1/5,
        )
