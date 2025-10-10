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
    data_definition(
        hyper_cols=['optimizer.alpha'],
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    
    columns = ["Environment", "Algorithm", "Hyperparameter", "Value", "AUC", "Return", "P_AUC"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, hyper, val, auc, ret, suc):
        new_entry = pd.DataFrame([[env, alg, hyper, val, auc, ret, suc]], columns=df.columns)
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
            
            alpha2auc = {}
            alpha2pauc = {}
            alpha2return = {}
            for alpha, alpha_df in df.group_by("optimizer.alpha"):
                alpha = alpha[0]

                xs, ys = extract_learning_curves(
                    alpha_df,
                    hyper_vals={"optimizer.alpha": alpha},
                    metric='reward',
                    interpolation=None,
                )

                auc = []
                p_auc = []
                total_reward = []
                for t, r in zip(xs, ys):
                    auc.append(np.mean(r))
                    p_auc.append(np.mean(np.split(np.array(r), 10)[-1])) # This also asserts 10% divisibility
                    total_reward.append(np.sum(r))
                alpha2auc[alpha] = np.mean(auc)
                alpha2pauc[alpha] = np.mean(p_auc)
                alpha2return[alpha] = np.mean(total_reward)
            
            best_alpha = max(alpha2auc, key=alpha2auc.get)
            best_auc = alpha2auc[best_alpha]
            best_p_auc = alpha2pauc[best_alpha]
            best_return = alpha2return[best_alpha]
            
            collector_df = add_entry(collector_df, env, alg, "optimizer.alpha", best_alpha, best_auc, best_return, best_p_auc)

    collector_df.to_csv(f"{path}/hyperparameter_collector.csv")