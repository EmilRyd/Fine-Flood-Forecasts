# evaluate and compare any list of models that you wish to evaluate

from pathlib import Path
import pandas as pd
import os
from neuralhydrology.nh_run import eval_run
import numpy as np
from experiment.utils import TrainedModel, TrainedModelID
# Display the DataFrame with HTML rendering
from IPython.core.display import display, HTML
import re

metric_goals = {
    'NSE': 1,
    'KGE': 1,
    'Alpha-NSE': 1,
    'Beta-NSE': 1,
    'MSE': 0,
    'RMSE': 0,
    'Pearson-r': 1,
    'Beta-KGE': 1,
    'FHV': 0,
    'FMS': 0,
    'FLV': 0,
    'Peak-Timing': 0,
    'Missed-Peaks': 0,
    'Peak-MAPE': 0
}
assets_dir = Path(__file__).parent / 'assets'
eval_dir = Path(__file__).parent / 'outputs' / 'evals'



def bold_better(row):

    # for a given metric (one row) returns a row with boldened text 
    # for the cell which has the best score
    vals = row.drop('Metric')

    stripped_metric = re.sub(r' \((mean|median)\)', '', row['Metric'])

    goal = metric_goals[stripped_metric]
    
    dists = [abs(val-goal) for val in vals]
    best_idx = np.argmin(dists)
    vals = [round(val, 2) for val in vals]
    
    new_row = [f"<b>{val}</b>" if idx==best_idx else val for idx, val in enumerate(vals)]
    return new_row


def evalute_model_csvs(model_eval_files: dict, basins: list, include_benchmark: bool, bolden_values: bool):
    """takes in a dict of model_name: eval_csvs pairs and generates a table comparing their metrics"""

    comparison_df = pd.DataFrame()
    for model, eval_file in model_eval_files.items():

        assert os.path.exists(eval_file), "Metric eval file does not exist. Did you run the eval script yet?"
        
        # read csv
        df = pd.read_csv(eval_file, dtype={'basin':str})

        if len(basins) > 0:
            df = df[df.basin.isin(basins)].reset_index(drop=True)

        df.drop(columns='basin', inplace=True)

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
        eval_df.columns = ['Metric', model] # set column labels

        if comparison_df.empty:
            comparison_df = eval_df
        else:
            comparison_df = pd.merge(left=comparison_df, right=eval_df, on='Metric')

    if include_benchmark:
        # read in benchmark
        bm_df = pd.read_csv(assets_dir / 'benchmark.csv', dtype={'Benchmark': float})
        comparison_df = pd.merge(left=comparison_df, right=bm_df, on='Metric')

    if bolden_values:
        # bolden best prediction
        model_keys = list(model_eval_files.keys())
        if include_benchmark:
            model_keys.append('Benchmark')
        comparison_df[model_keys] = comparison_df.apply(bold_better, axis=1, result_type='expand')

    return comparison_df



def evaluate_models(models: list, basins: list = [], include_benchmark: bool = True, period='test', bolden_values=False):
    """Takes list of TrainedModel objects and evalutes them against each other"""
    models_dict = {}
    for model in models:
        
        assert isinstance(model, TrainedModel), 'model is not a TrainedModel data object'

        # if any models are not evaluted yet, do so
        metrics_file = model.get_eval_metrics_file(period=period)
        if not os.path.exists(metrics_file):
            eval_run(model.run_dir, period=period, epoch=model.epoch)
        models_dict[model.config_id] = metrics_file
        
    # evalauate the model csvs
    df = evalute_model_csvs(models_dict, basins=basins, include_benchmark=include_benchmark, bolden_values=bolden_values)

    # write the evaluated df to disk
    df.to_csv(os.path.join(eval_dir, 'eval.csv'))

    return df


model = TrainedModel(TrainedModelID.SOTA_20)

evaluate_models([model], include_benchmark=False, period='validation')
