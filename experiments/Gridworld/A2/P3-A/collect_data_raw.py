import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
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
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    
    columns = ["Environment", "Algorithm", 
               "AUC"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, auc):
        new_entry = pd.DataFrame([[env, alg, auc]], columns=df.columns)
        return pd.concat([df, new_entry], ignore_index=True)

    for env, sub_results in results.groupby_directory(level=2):
        for alg_result in sub_results:
            alg = alg_result.filename
            if alg not in ["TMaze-DRQN-fta-use-all-steps-1-32", "TMaze-DRQN-use-all-steps-1-32"]:
                continue
            print(alg)

            df = alg_result.load()
            print(df)
            
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
            for t, r in zip(xs, ys):
                ave_r = []
                for i in range(int(total_steps/N)):
                    indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                    ave_r.append(np.mean(r[indices[-n:]]))
                auc.append(np.sum(ave_r).item())

            collector_df = add_entry(collector_df, env, alg, auc)

    collector_df.to_csv(f"{path}/learning_curve_raw.csv")