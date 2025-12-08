import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection(Model=ExperimentModel)
    hyper_cols = results.get_hyperparameter_columns()
    data_definition(
        hyper_cols=hyper_cols,
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    
    columns = ["Environment", "Algorithm", "AUC", "Return", "Successes"] + hyper_cols
    collector_df = pd.DataFrame(columns=columns)

    for env, sub_results in results.groupby_directory(level=4):
        for alg_result in sub_results:
            alg = alg_result.filename
            print(alg)

            df = alg_result.load()
            if df is None:
                continue

            exp = alg_result.exp
            total_steps = exp.total_steps
            
            active_hypers = [h for h in hyper_cols if h in df.columns]
            if not active_hypers:
                continue

            config2auc = {}
            config2return = {}
            for keys, group_df in df.group_by(active_hypers):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                
                current_hyper_vals = dict(zip(active_hypers, keys))

                xs, ys = extract_learning_curves(
                    group_df,
                    hyper_vals=current_hyper_vals,
                    metric='return',
                    interpolation=None,
                )
                
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
                config2auc[keys] = np.mean(auc)
                config2return[keys] = np.mean(total_reward)
            
            best_config = max(config2auc, key=config2auc.get)
            best_auc = config2auc[best_config]
            best_return = config2return[best_config]

            successes = []

            subset_df = df
            for col, val in zip(active_hypers, best_config):
                subset_df = subset_df.filter(pl.col(col) == val)

            for seed, seed_df in subset_df.group_by("seed"):
                seed_df = seed_df.filter(pl.col("frame") > total_steps * 0.9)
                successes.append(np.nanmean(seed_df['success'].to_numpy()))
            
            row_dict = {
                "Environment": env,
                "Algorithm": alg,
                "AUC": best_auc,
                "Return": best_return,
                "Successes": np.mean(successes),
            }
            for col, val in zip(active_hypers, best_config):
                row_dict[col] = val

            new_entry = pd.DataFrame([row_dict], columns=columns)
            print(new_entry)
            collector_df = pd.concat([collector_df, new_entry], ignore_index=True)

    collector_df.to_csv(f"{path}/hyperparameter_collector.csv")