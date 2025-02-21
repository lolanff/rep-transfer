import os
import sys
sys.path.append(os.getcwd() + '/src')
import json 
import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

from RlEvaluation.config import data_definition
from RlEvaluation.interpolation import compute_step_return
from RlEvaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from RlEvaluation.statistics import Statistic
from RlEvaluation.utils.pandas import split_over_column

import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel)

    data_definition(
        hyper_cols=['environment.goal_id'],
        seed_col='seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',
        make_global=True,
    )

    df = results.combine(
        folder_columns=(None, None, None, None, 'environment'),
        file_col='algorithm',
    )

    assert df is not None

    exp = results.get_any_exp()

    fig, ax = plt.subplots()
    for env, env_df in split_over_column(df, col='environment'):
        x_values = []
        y_values = []
        for alg, alg_df in split_over_column(env_df, col='algorithm'):    
            goal_id = alg_df['environment.goal_id'].unique()[0]
            xs, ys = extract_learning_curves(alg_df, (goal_id,), metric='return', interpolation=None)
            assert len(xs) == 5   # check all 5 seeds are there

            # Every N steps we record the average return of the last n episodes
            N = 10000
            n = 100
            auc = []
            for t, r in zip(xs, ys):
                ave_r = []
                for i in range(int(exp.total_steps/N)):
                    indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                    ave_r.append(np.mean(r[indices[-n:]]))
                auc.append(np.sum(ave_r))          

            x_values.append(int(goal_id))
            y_values.append(np.mean(auc))

        x_values = np.array(x_values)
        sorted_id = np.argsort(x_values)
        x_values = x_values[sorted_id]
        y_values = np.array(y_values)[sorted_id]

        ax.plot(x_values, y_values, label=env, linewidth=0.5)
        with open(f'./{env}.json', 'w') as f:
            json.dump(list(y_values), f)

    ax.set_ylim((0, 11))
    ax.legend()
    save(save_path=f'{path}/plots', plot_name=f'scratch_curves')
    plt.show()   