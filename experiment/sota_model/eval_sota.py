# evaluating the SOTA model from HESS paper by Kratzert et al (2020)
from pathlib import Path
import pandas as pd
import pickle as pkl
import glob
import os
from neuralhydrology.nh_run import eval_run
import matplotlib.pyplot as plt
import math

# Display the DataFrame with HTML rendering
from IPython.core.display import display, HTML

metric_goals = {
    'NSE (median)': 1,
    'NSE (mean)': 1,
    'KGE (median)': 1,
    'Pearson-r (median)': 1,
    'Alpha-NSE (median)': 1,
    'Beta-NSE (median)': 0,
    'FHV (median)': 0,
    'FLV (median)': 0,
    'FMS (median)': 0,
    'Peak-Timing (median)': 0
}
def bold_better(row):
    my_val = row['My Value']
    bm = row['Benchmark']

    goal = metric_goals[row['Metric']]
    
    my_dist = abs(my_val - goal)
    bm_dist = abs(bm - goal)

    my_val = round(my_val, 2)
    bm = round(bm, 2)

    if my_dist < bm_dist:
        return f"<b>{my_val}</b>", str(bm)
    else:
        return str(my_val), f"<b>{bm}</b>"
# path to eval data
run_dir = Path(__file__).parent / 'runs' / 'sota_20'
test_path = run_dir / 'test' / 'model_epoch030' / 'test_metrics.csv'

if not os.path.exists(test_path):
    eval_run(run_dir=run_dir, period='test', epoch=30)

# read csv
df = pd.read_csv(test_path).drop(columns='basin')

# average across the columns
mean_ = 'mean'
median_ = 'median'
ave_df = df.copy()
ave_df.columns = [i + ' (' + mean_ + ')' for i in df.columns]
ave_df = ave_df.agg('mean', axis=0)
median_df = df.copy()
median_df.columns = [i + ' (' + median_ + ')' for i in df.columns]
median_df = median_df.agg('median', axis=0)
eval_df = pd.concat((ave_df, median_df)).reset_index(drop=False)
eval_df.columns = ['Metric', 'My Value'] # set column labels

# read in benchmark
bm_df = pd.read_csv('benchmark.csv', dtype={'Benchmark': float})


comparison_df = pd.merge(left=bm_df, right=eval_df, on='Metric')

# bolden best prediction
comparison_df[['My Value', 'Benchmark']] = comparison_df.apply(bold_better, axis=1, result_type='expand')

display(HTML(comparison_df.to_html(escape=False)))
