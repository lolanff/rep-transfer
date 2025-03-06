import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
import pickle
import lzma
from experiment.tools import parseCmdLineArgs
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


setDefaultConference('jmlr')
path, should_save, save_type = parseCmdLineArgs()

# Load the data
file_path = f'{path}/results.csv'
data = pd.read_csv(file_path, index_col=0)

# Custom Radar Factory
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

# Define metrics and labels
metrics = ['Complexity Reduction', 'Diversity', 'Dynamics Awareness', 'Orthogonality', 'Sparsity']
labels = metrics + metrics[:1]  # Close the loop for labels
num_vars = len(metrics)
theta = radar_factory(num_vars, frame='polygon')

# Prepare data for the two algorithms
# Rename the labels: DQN-ReLU-A -> No Aux, DQNAux-ReLU-A -> VF5
algorithms = ['DQN-ReLU-A', 'DQNAux-ReLU-A']
new_labels = {'DQN-ReLU-A': 'No Aux', 'DQNAux-ReLU-A': 'VF5'}
data_mean = {new_labels[alg]: data[data['algorithm'] == alg][metrics].mean().values for alg in algorithms}

# Plot the radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
fig.suptitle("Representation Properties", fontsize=28, fontweight='bold')

# Plot each algorithm
colors = ['blue', 'red']
for i, (algorithm, values) in enumerate(data_mean.items()):
    values = np.append(values, values[0])  # Close the loop for values
    ax.plot(theta, values, color=colors[i], label=algorithm, linewidth=2)
    ax.fill(theta, values, color=colors[i], alpha=0.25)

# Set labels and other settings
ax.set_varlabels(labels)  # Use the closed loop labels

# Customize Legend
legend = plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=22, frameon=True, fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontsize(22)

plt.savefig(f'{path}/rep_prop.png')
