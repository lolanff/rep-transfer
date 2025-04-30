import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import bootstrap
import pandas as pd
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

ci_percentile = 0.95

def compute_bootstrap_ci(data, ci=0.95):
    if len(data) > 1:
        ci_bounds = bootstrap((data,), np.mean, confidence_level=ci, n_resamples=10000, method='percentile').confidence_interval
        return ci_bounds.low, ci_bounds.high
    else:
        return data[0], data[0]

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection(Model=ExperimentModel)
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    
    columns = ["Environment", "Algorithm", 
               "AUC_early", "AUC_early_low", "AUC_early_high",
               "AUC_late", "AUC_late_low", "AUC_late_high", 
               "AUC", "AUC_low", "AUC_high"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, 
                  auc_early, auc_early_low, auc_early_high,
                  auc_late, auc_late_low, auc_late_high,
                  auc, auc_low, auc_high):
        new_entry = pd.DataFrame([[env, alg, 
                                   auc_early, auc_early_low, auc_early_high,
                                   auc_late, auc_late_low, auc_late_high,
                                   auc, auc_low, auc_high]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)

    for env, sub_results in results.groupby_directory(level=4):
        for alg_result in sub_results:
            alg = alg_result.filename
            print(alg)

            df = alg_result.load()
            if df is None:
                continue

            exp = alg_result.exp
            total_steps = exp.total_steps

            xs, ys = extract_learning_curves(
                df,
                hyper_vals={},
                metric='return',
                interpolation=None,
            )
                
            # Every N steps we record the average return of the last n episodes
            N = 10_000
            n = 100
            auc = []
            auc_early = []
            auc_late = []
            for t, r in zip(xs, ys):
                ave_r = []
                for i in range(int(total_steps/N)):
                    indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                    ave_r.append(np.mean(r[indices[-n:]]))
                    
                sums = np.sum(np.split(np.array(ave_r), 2), axis=1)
                auc_early.append(sums[0])
                auc_late.append(sums[1])
                auc.append(np.sum(ave_r))

            auc_early_mean = np.mean(auc_early)
            auc_late_mean = np.mean(auc_late)
            auc_mean = np.mean(auc)
            auc_early_lower, auc_early_upper = compute_bootstrap_ci(auc_early, ci=0.95)
            auc_late_lower, auc_late_upper = compute_bootstrap_ci(auc_late, ci=0.95)
            auc_lower, auc_upper = compute_bootstrap_ci(auc, ci=0.95)

            collector_df = add_entry(
                collector_df, env, alg, 
                auc_early_mean ,auc_early_lower, auc_early_upper,
                auc_late_mean, auc_late_lower, auc_late_upper, 
                auc_mean, auc_lower, auc_upper)

    collector_df.to_csv(f"{path}/collector.csv")