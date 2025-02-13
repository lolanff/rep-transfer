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

THIS_AGENT = 'DQN-FTA-sweep'

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
        folder_columns=(None, None, None, 'environment'),
        file_col='algorithm',
    )

    assert df is not None

    exp = results.get_any_exp()

    for env, env_df in split_over_column(df, col='environment'):
        for alg, alg_df in split_over_column(env_df, col='algorithm'):            
            if alg == THIS_AGENT: 
                best_alpha = {}
                for goal_id, goal_df in alg_df.groupby('environment.goal_id'):
                    alpha2auc = {}
                    for alpha in goal_df['optimizer.alpha'].unique():
                        xs, ys = extract_learning_curves(goal_df, (alpha,), metric='return', interpolation=None)
                
                        # Every N steps we record the average return of the last n episodes
                        N = 10000
                        n = 100
                        auc = []
                        for t, r in zip(xs, ys):
                            ave_r = []
                            for i in range(int(exp.total_steps/N)):
                                indices = np.where((N*i < t) & (t <= N*(i+1)))[0]
                                ave_r.append(np.mean(r[indices[-n:]]))
                            auc.append(np.sum(ave_r))
                        alpha2auc[alpha] = np.mean(auc)
                    
                    best_alpha[goal_id] = max(alpha2auc, key=alpha2auc.get)
                        
                print(best_alpha)          
                x = list(map(int, best_alpha.keys()))
                y = list(best_alpha.values())  
                plt.plot(x, y, '.')     
                plt.show()   