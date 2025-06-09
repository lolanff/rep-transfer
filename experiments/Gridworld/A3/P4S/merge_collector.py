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

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()
    
    paths = [f"{path}/../P4/collector.csv", f"{path}/collector.csv"]
    
    combined_df = pd.concat([pd.read_csv(path, index_col=0) for path in paths], axis=0, ignore_index=True)

    combined_df.to_csv(f"{path}/combined_collector.csv")