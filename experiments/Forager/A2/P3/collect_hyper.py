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
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    HYPER_COLS = [
        "optimizer.eps",
        "optimizer.alpha",
        "update_freq",
        "target_refresh",
        "optimizer.beta2",
    ]

    columns = ["Environment", "Algorithm", "AUC"] + HYPER_COLS
    collector_df = pd.DataFrame(columns=columns)

    def add_entry(df, env, alg, hyper_dict, auc):
        hyper_vals = [hyper_dict.get(col) for col in HYPER_COLS]
        row = [env, alg, auc] + hyper_vals
        new_entry = pd.DataFrame([row], columns=df.columns)
        print(new_entry)
        return pd.concat([df, new_entry], ignore_index=True)

    for env, sub_results in results.groupby_directory(level=3):
        for alg_result in sub_results:
            alg = alg_result.filename
            print(alg)

            df = alg_result.load()
            if df is None:
                continue

            report = Hypers.select_best_hypers(
                df,
                metric='reward',
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.mean,
                statistic=Statistic.mean,
            )
            
            filtered = {k: v for k, v in report.best_configuration.items() if k in HYPER_COLS}
            collector_df = add_entry(collector_df, env, alg, filtered, report.best_score)

    collector_df.to_csv(f"{path}/hyperparameter_collector.csv")