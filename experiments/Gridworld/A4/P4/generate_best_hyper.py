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
    
    old_path = f"{path}/../P3"
    
    df = pd.read_csv(f"{old_path}/hyperparameter_collector.csv")
    
    files = []

    for _, row in df.iterrows():
        env = row['Environment']
        alg = row['Algorithm']
        val = row['Value']
        goal = str(row['Goal'])

        json_input_path = f"{old_path}/{env}/{alg}.json"
        env = env.replace("_hyper_sweep", "")
        alg = alg.replace("-hyper-sweep", "")
        json_output_path = f"{path}/{env}/{alg}.json"

        if not os.path.isfile(json_input_path):
            print(f"Warning: JSON file not found: {json_input_path}")
            continue

        with open(json_input_path, 'r') as f:
            data = json.load(f)

        data['metaParameters']['optimizer']['alpha'] = val
        data['metaParameters']['experiment']['seed_offset'] += 10000
        data['metaParameters']['experiment']['save'] = False

        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Generated: {json_output_path}")
        
        files.append(json_output_path)
        
    record_path = f"{path}/record.txt"
    with open(record_path, "w") as f:
        print(" ".join(map(str, files)), file=f)