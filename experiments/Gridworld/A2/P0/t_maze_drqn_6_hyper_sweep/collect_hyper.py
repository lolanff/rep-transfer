import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return
from rlevaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from rlevaluation.statistics import Statistic

import rlevaluation.hypers as Hypers
import rlevaluation.metrics as Metrics

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
    
    exp = results.get_any_exp()
    total_steps = exp.total_steps

    del results
    assert df is not None
    
    df.to_csv(f"{path}/collector.csv")
    
    columns = ["Environment", "Algorithm", "Hyperparameter", "Value", "AUC", "Return", "Successes"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, hyper, val, auc, ret, suc):
        new_entry = pd.DataFrame([[env, alg, hyper, val, auc, ret, suc]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)

    for env, env_df in df.groupby(by='environment'):
        for alg, alg_df in env_df.groupby(by='algorithm'):
            alpha2auc = {}
            alpha2return = {}
            for alpha, alpha_df in alg_df.groupby(by='optimizer.alpha'):
                xs, ys = extract_learning_curves(alpha_df, (alpha,), metric='return', interpolation=None)
            
                # Every N steps we record the average return of the last n episodes
                N = 10_000
                n = 100
                auc = []
                total_reward = []
                for t, r in zip(xs, ys):
                    ave_r = []
                    for i in range(int(total_steps/N)):
                        indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                        ave_r.append(np.mean(r[indices[-n:]]))
                    auc.append(np.sum(ave_r))
                    total_reward.append(np.sum(r))
                alpha2auc[alpha] = np.mean(auc)
                alpha2return[alpha] = np.mean(total_reward)
            
            best_alpha = max(alpha2auc, key=alpha2auc.get)
            best_auc = alpha2auc[best_alpha]
            best_return = alpha2return[best_alpha]
            
            successes = []

            for seed, seed_df in alg_df[df['optimizer.alpha']==best_alpha].groupby('seed'):
                seed_df = seed_df[seed_df['frame'] > total_steps * 0.9]
                successes.append(np.nanmean(seed_df['success'].values))
            
            collector_df = add_entry(collector_df, env, alg, "optimizer.alpha", best_alpha, best_auc, best_return, np.mean(successes))

    collector_df.to_csv(f"{path}/hyperparameter_collector.csv")
