# perform finetuning on specific basin
# find optimal hyperparameters for finetuning
# Imports
from pathlib import Path
import tempfile
import json
from experiment.utils import TrainedModel
import pandas as pd
from neuralhydrology.nh_run import finetune
from experiment.eval import evaluate_models
import os
import yaml
import numpy as np

from hyperopt import fmin, Trials, tpe, hp



# pick a basin
def pick_a_basin(model: TrainedModel):
    df = pd.read_csv(model.get_eval_metrics_file(period='validation'), dtype={'basin':str})
    cutoff = 0.0
    basin_data = df.loc[df['NSE'] > cutoff].sample(n=1)
    basin = basin_data.basin.iloc[0]
    nse = basin_data.NSE.iloc[0]
    return basin, nse

def update_files(model: TrainedModel, basin: str, yml_file_path="finetune.yml"):
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

def finetune_model(args):

    basin = args['basin']
    model = args['model']
    
    # get base cfg
    yml_file_path = Path(__file__).parent / 'finetune.yml'

   # Load the existing YAML data
    with open(yml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    # set dict parameters based on config dictionary passed to function
    modules = ['head']
    for key, value in args.items():
        if not (key == 'basin') and not (key == 'model'):
            if (key == 'lstm'):
                if value:
                    modules.append(key)
            else:
                data[key] = value

    data['finetune_modules'] = modules
    # finetune using temporary yaml file
    #
    with tempfile.NamedTemporaryFile(delete=True, dir=Path(__file__).parent / 'assets', suffix='.yml', mode='w') as f:
        yaml.dump(data, f)  

        finetune(Path(__file__).parent / 'assets' / f.name)

        run_dir = Path(os.path.abspath('')) / 'runs' / f'basin_{basin}'
        config_file_path = run_dir / 'config.yml'

        finetuned_model = TrainedModel(config_file_path_or_experiment_name=config_file_path)

        # TODO validation returns all nans!
        t_df = evaluate_models([model, finetuned_model], basins=[basin], include_benchmark=False, period='train')
        v_df = evaluate_models([model, finetuned_model], basins=[basin], include_benchmark=False, period='validation')

    return {'loss': -float(v_df.iloc[0][f'basin_{basin}']), 'model': finetuned_model}


def find_best_finetuning_params(basin: str, search_space: dict):
   
    
    trials = Trials()
    best = fmin(finetune_model, space=search_space, algo=tpe.suggest, max_evals=3, trials=trials)
    
    return best

    
