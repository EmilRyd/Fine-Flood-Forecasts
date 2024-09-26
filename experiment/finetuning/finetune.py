# perform finetuning on specific basin
# find optimal hyperparameters for finetuning
# Imports
from pathlib import Path
import tempfile
from experiment.utils import TrainedModel
import pandas as pd
from neuralhydrology.nh_run import finetune
from experiment.eval import evaluate_models
import os
import yaml
from experiment.finetuning.utils import Sweep, param_dict_from_model_output
import logging
from hyperopt import fmin, Trials, tpe, STATUS_OK
from datetime import datetime


# pick a basin
def pick_a_basin(model: TrainedModel, lower: float = None, higher: float = None):
    df = pd.read_csv(model.get_eval_metrics_file(period='validation'), dtype={'basin':str})
    c_time = int(datetime.now().timestamp())
    if lower is None or higher is None:
        basin_data = df.sample(n=1, random_state=c_time)
    else:
        basin_data = df.loc[(df['NSE'] <= higher) & (df['NSE'] >= lower)].sample(n=1, random_state=c_time)

    basin = basin_data.basin.iloc[0]
    nse = basin_data.NSE.iloc[0]
    update_files(model=model, basin=basin)
    return basin, nse

def update_files(model: TrainedModel, basin: str, yml_file_path=os.path.join('assets', 'finetune.yml')):
    
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

    basin_file_path = data['train_basin_file']
    
    # Create a basin file with the basin we selected above
    with open(basin_file_path, 'w') as fp:
        fp.write(basin) 

def finetune_model(args):

    basin = args['basin']

    # sanity check on args
    assert args['epochs'] > 0, f"Number of epochs is invalid: {args['epochs']}"
    # get base cfg
    yml_file_path = Path(__file__).parent / 'assets' / 'finetune.yml'

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
            elif (key == 'epochs'):
                data[key] = int(value) # make this cleaner mayber
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

        v_df = evaluate_models([finetuned_model], basins=[basin], include_benchmark=False, period='validation', ignore_previous_metrics=True)
    
    # return negative validation score
    return {'loss': -float(v_df.iloc[0][f'basin_{basin}']), 'status': STATUS_OK, 'model': finetuned_model}



def find_best_finetuning_params(search_space: dict, model: TrainedModel, max_evals=10, evaluate=True) -> Sweep:
   
    trials = Trials()
    best_params = fmin(finetune_model, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # add basin back to best params
    args = param_dict_from_model_output(best_params, search_space['basin'])

    finetuning_data = finetune_model(args)
    finetuned_model = finetuning_data['model']

    # store model, finetuned model, and best_params
    sweep = Sweep(best_params=best_params, base_model=model, finetuned_model=finetuned_model, search_space=search_space, max_evals=max_evals)

    # perform evaluation on all basins
    if evaluate:
        evaluate_models(models=[sweep.finetuned_model], bolden_values=True, include_benchmark=False, ignore_previous_metrics=True)
    
    return sweep