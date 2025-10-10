import json
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
    
    HYPER_COLS = [
        "optimizer.eps",
        "optimizer.alpha",
        "update_freq",
        "target_refresh",
        "optimizer.beta2",
    ]
    
    old_path = f"{path}/../P3"
    
    df = pd.read_csv(f"{old_path}/hyperparameter_collector.csv")
    
    files = []

    for _, row in df.iterrows():
        env = row['Environment']
        alg = row['Algorithm']

        json_input_path = f"{old_path}/{alg}.json"
        json_output_path = f"{path}/{alg}.json"

        if not os.path.isfile(json_input_path):
            print(f"Warning: JSON file not found: {json_input_path}")
            continue

        with open(json_input_path, 'r') as f:
            data = json.load(f)

        metaParameters = data['metaParameters']
        for hyper in HYPER_COLS:
            val = row[hyper]
            curMetaParameter = metaParameters
            keys = hyper.split('.')
            for key in keys[:-1]:
                curMetaParameter = curMetaParameter[key]
            curMetaParameter[keys[-1]] = val
        metaParameters['experiment']['seed_offset'] += 10000
        data['total_steps'] = 500000
        metaParameters['epsilon_steps'] = 400000

        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Generated: {json_output_path}")
        
        files.append(json_output_path)
        
    record_path = f"{path}/record.txt"
    with open(record_path, "w") as f:
        print(" ".join(map(str, files)), file=f)