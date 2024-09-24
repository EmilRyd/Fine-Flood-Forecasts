# perform finetuning on specific basin
# find optimal hyperparameters for finetuning
# Imports
from pathlib import Path
import tempfile
from experiment.utils import TrainedModel, TrainedModelID, get_losses

import pandas as pd
import torch
from neuralhydrology.nh_run import start_run, eval_run, finetune
from neuralhydrology.utils.config import Config
from experiment.eval import evaluate_models
import os
import yaml
import matplotlib.pyplot as plt
import optuna
import numpy as np

# start with SOTA Camels model
model = TrainedModel(TrainedModelID.SOTA_20)

# pick a basin
def pick_a_basin():
    df = pd.read_csv(model.get_eval_metrics_file(period='validation'), dtype={'basin':str})
    cutoff = 0.0
    basin_data = df.loc[df['NSE'] > cutoff].sample(n=1)
    basin = basin_data.basin.iloc[0]
    nse = basin_data.NSE.iloc[0]
    return basin, nse

def update_files(basin, yml_file_path="finetune.yml"):
    # Add the path to the pre-trained model to the finetune config

    with open(yml_file_path, "a") as fp:
        fp.write(f"\nbase_run_dir: {model.run_dir.absolute()}")

    # Load the existing YAML data
    with open(yml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    data['experiment_name'] = f'basin_{basin}'  # Example modification

    # Write back to the YAML file
    with open(yml_file_path, 'w') as f:
        yaml.dump(data, f)  

    # Create a basin file with the basin we selected above
    with open(f"finetune_basin.txt", "w") as fp:
        fp.write(basin) 

def finetune_model(trial):

   # Load the existing YAML data
    with open('finetune.yml', 'r') as f:
        data = yaml.safe_load(f)
    
    epochs = trial.suggest_int('epochs', 1, 3)
    
    # set dict parameters based on config dictionary passed to function    
    data['epochs'] = epochs
    
    # finetune using temporary yaml file
    
    with tempfile.NamedTemporaryFile(delete=True, dir=Path(__file__).parent, suffix='.yml', mode='w') as f:
        yaml.dump(data, f)  
        print(f.name)
        finetune(Path(__file__).parent / f.name)

        run_dir = Path(os.path.abspath('')) / 'runs' / f'basin_{basin}'
        config_file_path = run_dir / 'config.yml'

        finetuned_model = TrainedModel(config_file_path_or_experiment_name=config_file_path)
        
        t_df = evaluate_models([model, finetuned_model], basins=[basin], include_benchmark=False, period='train')
        v_df = evaluate_models([model, finetuned_model], basins=[basin], include_benchmark=False, period='validation')
        trial.report(float(v_df.iloc[0][f'basin_{basin}']), epochs)
        return float(v_df.iloc[0][f'basin_{basin}'])



# pick a basin and update config file accordingly
basin, nse = pick_a_basin()

# update config yml and basin txt files
update_files(basin=basin)

# perform hyperparameter search over finetuning
study = optuna.create_study()
study.optimize(finetune_model, n_trials=1)














