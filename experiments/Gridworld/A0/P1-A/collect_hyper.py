import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import pandas as pd
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
        hyper_cols=['optimizer.alpha'],
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
    
    columns = ["Environment", "Algorithm", "Goal", "Hyperparameter", "Value", "AUC", "Return"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, goal, hyper, val, auc, ret):
        new_entry = pd.DataFrame([[env, alg, goal, hyper, val, auc, ret]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        for alg, alg_df in split_over_column(env_df, col='algorithm'):
            best_auc = {}
            best_return = {}
            for goal_id, goal_df in alg_df.groupby('environment.goal_id'):
                alpha2auc = {}
                alpha2return = {}
                for alpha in goal_df['optimizer.alpha'].unique():
                    xs, ys = extract_learning_curves(goal_df, (alpha,), metric='return', interpolation=None)
              
                    # Every N steps we record the average return of the last n episodes
                    N = 10_000
                    n = 100
                    auc = []
                    total_reward = []
                    for t, r in zip(xs, ys):
                        ave_r = []
                        for i in range(int(exp.total_steps/N)):
                            indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                            ave_r.append(np.mean(r[indices[-n:]]))
                        auc.append(np.sum(ave_r))
                        total_reward.append(np.sum(r))
                    alpha2auc[alpha] = np.mean(auc)
                    alpha2return[alpha] = np.mean(total_reward)
                
                best_alpha = max(alpha2auc, key=alpha2auc.get)
                best_auc[goal_id] = alpha2auc[best_alpha]
                best_return[goal_id] = alpha2return[best_alpha]
                collector_df = add_entry(collector_df, env, alg, goal_id, "optimizer.alpha", best_alpha, best_auc[goal_id], best_return[goal_id])
                
    collector_df.to_csv(f"{path}/hyperparameter_collector.csv")
