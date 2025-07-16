import os
import sys
sys.path.append(os.getcwd() + '/src')
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from experiment.tools import parseCmdLineArgs
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

from PyExpPlotting.matplot import save, setDefaultConference

setDefaultConference('neurips')

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

    df = pd.read_csv(f'{path}/learning_curve.csv', index_col=0)

    for col in ['X', 'Y', 'Low', 'High']:
        df[col] = (
            df[col]
            .str.strip('[]')
            .str.split()
            .apply(lambda lst: [float(item) for item in lst])
        )

    algorithm_name_map = {
        'Gridworld-DQN-fta': 'DQN FTA (BS=32 SL=1)',
        'Gridworld-DQN': 'DQN ReLU (BS=32 SL=1)',
        'Gridworld-DRQN-fta': 'DRQN FTA (BS=2 SL=16)',
        'Gridworld-DRQN': 'DRQN ReLU (BS=2 SL=16)'
    }

    df['AlgorithmMapped'] = df['Algorithm'].map(algorithm_name_map).fillna(df['Algorithm'])
    
    fig, ax = plt.subplots(1, 1)
    
    for index, row in df.iterrows():
        x_vals = np.array(row['X'])
        y_vals = np.array(row['Y'])
        low_vals = np.array(row['Low'])
        high_vals = np.array(row['High'])

        label = row['AlgorithmMapped']
        ax.plot(x_vals, y_vals, linewidth=1.0, label=label)
        ax.fill_between(x_vals, low_vals, high_vals, alpha=0.2)

    ax.set_xticks(np.arange(0, 100000 + 1, 25000))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend()

    save(
        save_path=f'{path}/plots',
        plot_name='learning_curve',
        f=fig,
        height_ratio=2/3,
    )