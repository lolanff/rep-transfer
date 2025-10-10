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
from tqdm import tqdm

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
    data_definition(
        hyper_cols={},
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    
    columns = ["Environment", "Algorithm", "Goal", "Steps", "Run", "AUC", "Return", "Successes"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, goal, steps, run, auc, ret, suc):
        new_entry = pd.DataFrame([[env, alg, goal, steps, run, auc, ret, suc]], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)

    for env, sub_results in tqdm(results.groupby_directory(level=4)):
        for alg_result in tqdm(sub_results):
            alg = alg_result.filename
            print(alg)

            df = alg_result.load()
            if df is None:
                continue

            exp = alg_result.exp
            total_steps = exp.total_steps
            
            for goal, goal_df in tqdm(df.group_by("environment.goal_id")):
                goal = goal[0]
                for seed, seed_df in goal_df.group_by("seed"):
                    seed = seed[0]

                    xs, ys = extract_learning_curves(
                        seed_df,
                        hyper_vals={},
                        metric='return',
                        interpolation=None,
                    )
                    
                    # Every N steps we record the average return of the last n episodes
                    N = 10_000
                    n = 100
                    auc = None
                    total_reward = None
                    for t, r in zip(xs, ys):
                        ave_r = []
                        for i in range(int(total_steps/N)):
                            indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                            ave_r.append(np.mean(r[indices[-n:]]))
                        auc = np.sum(ave_r)
                        total_reward = np.sum(r)

                    seed_df = seed_df.filter(pl.col("frame") > total_steps * 0.9)
                    successes = np.nanmean(seed_df['success'].to_numpy())
                
                    collector_df = add_entry(collector_df, env, alg, goal, total_steps, seed, auc, total_reward, successes)

    collector_df.to_csv(f"{path}/collector.csv")