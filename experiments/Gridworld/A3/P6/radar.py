import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/src')

from PyExpPlotting.matplot import save, setDefaultConference
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
import pickle
import lzma
from experiment.tools import parseCmdLineArgs
import jax.numpy as jnp
import time
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from scipy.stats import bootstrap  # Import the bootstrap function from SciPy

setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

file_path = f'{path}/combined_results.csv'
data = pd.read_csv(file_path, index_col=0)

# Custom Radar Factory adapted from https://github.com/erfanMhi/LTA-Representation-Properties.git
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    theta += theta[:1]  # Close the loop for theta

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if len(x) == 0 or len(y) == 0:
                line.set_data(x, y)
            elif x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=18)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(
                    Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    return theta

def compute_bootstrap_ci(data, ci=0.95):
    if len(data) > 1:
        ci_bounds = bootstrap((data,), np.mean, confidence_level=ci, n_resamples=1000, method='percentile').confidence_interval
        return ci_bounds.low, ci_bounds.high
    else:
        return data[0], data[0]  # Return the same value if only one data point exists

metrics = ['Complexity Reduction', 'Diversity', 'Dynamics Awareness', 'Orthogonality', 'Sparsity']
labels = metrics + metrics[:1]  # Close the loop for labels
num_vars = len(metrics)
theta = radar_factory(num_vars, frame='polygon')

config = {
    'ReLU': {
        'algorithm': {
            'DQN-ReLU-A': 'red',
            'DRQN-ReLU-Gridworld': 'blue',
        },
        'new_labels': {
            'DQN-ReLU-A': 'FF ReLU', 
            'DRQN-ReLU-Gridworld': 'RNN ReLU', 
        }
    },
    'ReLU (VF)': {
        'algorithm': {
            'DQNAux-ReLU-A': 'red',
            'DRQNAux-ReLU-Gridworld': 'blue',
        },
        'new_labels': {
            'DQNAux-ReLU-A': 'FF ReLU (VF1)', 
            'DRQNAux-ReLU-Gridworld': 'RNN ReLU (VF5)'
        }
    },
    'FTA': {
        'algorithm': {
            'DQN-FTA-A': 'red',
            'DRQN-FTA-Gridworld': 'blue',
            # 'DRQNAux-FTA-Gridworld': 'yellow'
        },
        'new_labels': { 
            'DQN-FTA-A': 'FF FTA',
            'DRQN-FTA-Gridworld': 'RNN FTA',
            # 'DRQNAux-FTA-Gridworld': 'RNN FTA (VF5)'
        }
    }
}

for name, info in config.items():
    algorithms = info['algorithm']
    new_labels = info['new_labels']
    
    data_mean = {}
    data_err_lower = {}
    data_err_upper = {}

    for alg, color in algorithms.items():
        alg_label = new_labels[alg]
        metric_means = []
        metric_err_lower = []
        metric_err_upper = []
        subset = data[data['algorithm'] == alg]
        for metric in metrics:
            values = subset[metric].values
            mean_val = np.mean(values)
            low, high = compute_bootstrap_ci(values)
            metric_means.append(mean_val)
            metric_err_lower.append(mean_val - low)
            metric_err_upper.append(high - mean_val)
        data_mean[alg_label] = np.array(metric_means)
        data_err_lower[alg_label] = np.array(metric_err_lower)
        data_err_upper[alg_label] = np.array(metric_err_upper)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    fig.suptitle(f"Representation Properties {name}", fontsize=28, fontweight='bold')

    for alg, color in algorithms.items():
        label = new_labels[alg]
        means = data_mean[label]
        lower_errors = data_err_lower[label]
        upper_errors = data_err_upper[label]
        # Close the loop by appending the first element at the end
        means_closed = np.append(means, means[0])
        lower_errors_closed = np.append(lower_errors, lower_errors[0])
        upper_errors_closed = np.append(upper_errors, upper_errors[0])
        
        ax.plot(theta, means_closed, color=color, label=label, linewidth=2)
        ax.fill(theta, means_closed, color=color, alpha=0.25)
        
        ax.errorbar(theta, means_closed, yerr=[lower_errors_closed, upper_errors_closed],
                    fmt='none', ecolor=color, capsize=5, elinewidth=2)

    # Set labels for each axis
    ax.set_varlabels(labels)

    legend = plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=22,
                        frameon=True, fancybox=True, shadow=True)
    for text in legend.get_texts():
        text.set_fontsize(22)

    plt.savefig(f'{path}/rep_prop_{name}.png')