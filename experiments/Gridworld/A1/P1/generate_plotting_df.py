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
        hyper_cols=results.get_hyperparameter_columns(),
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
    
    exp = results.get_any_exp()
    total_steps = exp.total_steps

    del results
    assert df is not None
    
    columns = ["Environment", "Algorithm", "Goal", "Seed", "AUC"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, goal, seed, auc):
        new_entry = pd.DataFrame([[env, alg, goal, seed, auc]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)

    N = 10_000
    n = 100

    for env, env_df in split_over_column(df, col='environment'):
        for alg, alg_df in split_over_column(env_df, col='algorithm'):
            for goal_id, goal_df in split_over_column(alg_df, col='environment.goal_id'):
                for seed, seed_df in split_over_column(goal_df, col='seed'):
                    xs, ys = extract_learning_curves(seed_df, (), metric='return', interpolation=None)
                    ave_r = []
                    for t, r in zip(xs, ys):
                        for i in range(int(total_steps/N)):
                            indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                            ave_r.append(np.mean(r[indices[-n:]]))
                    auc = np.sum(ave_r)
                    collector_df = add_entry(collector_df, env, alg, goal_id, seed, auc)
                
    collector_df.to_csv(f"{path}/collector.csv")
