import pandas as pd
import matplotlib.pyplot as plt
import ast
import re
import seaborn as sns

# Load the combined CSV
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "combined_collector.csv")
df = pd.read_csv(csv_path)

df["Algorithm"] = df["Algorithm"].str.replace("-hyper-sweep", "", regex=False)

# Plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Algorithm", y="Successes")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.title("Success Rate Distribution by Algorithm (Last 10% of training)")
plt.tight_layout()
plt.show()