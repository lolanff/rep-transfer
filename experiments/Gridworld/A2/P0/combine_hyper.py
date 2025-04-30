import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "combined_collector.csv")

combined_df = pd.DataFrame()

for root, dirs, files in os.walk(script_dir):
    if "hyperparameter_collector.csv" in files:
        file_path = os.path.join(root, "hyperparameter_collector.csv")
        print(f"Loading: {file_path}")
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.to_csv(output_file, index=False)