import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
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

        # makes this data definition globally accessible
        # so we don't need to supply it to all API calls
        make_global=True,
    )

    df = results.combine(
        # converts path like "experiments/example/MountainCar"
        # into a new column "environment" with value "MountainCar"
        # None means to ignore a path part
        folder_columns=(None, None, None, 'environment'),

        # and creates a new column named "algorithm"
        # whose value is the name of an experiment file, minus extension.
        # For instance, ESARSA.json becomes ESARSA
        file_col='algorithm',
    )

    assert df is not None
    Metrics.add_step_weighted_return(df)

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        for alg, sub_df in split_over_column(env_df, col='algorithm'):
            if len(sub_df) == 0: continue

            f, ax = plt.subplots()            
            LABELS = ['best alpha', 'second best alpha']
            COLORS = ['red', 'green']
            for alpha_id in range(2):  # Loop through best and second best alphas
                report = Hypers.select_best_hypers(
                    sub_df,
                    metric='step_weighted_return',
                    prefer=Hypers.Preference.high,
                    time_summary=TimeSummary.sum,
                    statistic=Statistic.mean,
                )

                print('-' * 25)
                print(env, alg)
                Hypers.pretty_print(report)

                xs, ys = extract_learning_curves(
                    sub_df,
                    report.best_configuration,
                    metric='return',
                    interpolation=None,
                )

                # Every N steps we record the average return of the last n episodes
                N = 10_000
                n = 100
                new_ys = []
                for t, r in zip(xs, ys):
                    ave_r = []
                    for i in range(int(exp.total_steps/N)):
                        indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                        ave_r.append(np.mean(r[indices[-n:]]))
                    new_ys.append(ave_r)
                ys = np.asarray(new_ys)
                xs0 = [N*(i+1) for i in range(int(exp.total_steps/N))]

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                )

                current_best_alpha = dict(zip(report.config_params, report.best_configuration)).get('optimizer.alpha')
                
                ax.plot(xs0, res.sample_stat, label=f"{LABELS[alpha_id]} {current_best_alpha}", color=COLORS[alpha_id], linewidth=0.5)
                ax.fill_between(xs0, res.ci[0], res.ci[1], color=COLORS[alpha_id], alpha=0.2)

                # Delete rows in dataframe corresponding to the previous best alpha
                sub_df = sub_df[sub_df['optimizer.alpha'] != current_best_alpha]

            ax.legend()

            save(
                save_path=f'{path}/plots',
                plot_name=f'{env}-{alg}'
            )
            plt.clf()

