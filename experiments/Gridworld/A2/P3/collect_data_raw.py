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
               "X", "Y"]
    collector_df = pd.DataFrame(columns=columns)
    
    def add_entry(df, env, alg, x, y):
        new_entry = pd.DataFrame([[env, alg, x, y]], columns=df.columns)
        return pd.concat([df, new_entry], ignore_index=True)

    for env, sub_results in results.groupby_directory(level=2):
        for alg_result in sub_results:
            alg = alg_result.filename
            if alg not in ["TMaze-DRQN-fta-use-all-steps-1-32"]:
                continue
            print(alg)

            df = alg_result.load()
            if df is None:
                continue

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals={},
                metric='return',
                interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
            )

            xs = np.asarray(xs)[:, ::exp.total_steps // 1000]
            ys = np.asarray(ys)[:, ::exp.total_steps // 1000]
            assert np.all(np.isclose(xs[0], xs))
            
            
            for x, y in zip(xs, ys):
                collector_df = add_entry(collector_df, env, alg, x, y)

        collector_df.to_csv(f"{path}/learning_curve_raw.csv")