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
from experiment.finetuning.utils import Sweep, cfg_from_args
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
    data = cfg_from_args(args)
    
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
    finetuned_model = trials.best_trial['result']['model']

    # store model, finetuned model, and best_params
    sweep = Sweep(best_params=best_params, base_model=model, 
                  finetuned_model=finetuned_model, search_space=search_space,
                    max_evals=max_evals, trials=trials)

    # perform evaluation on all basins
    if evaluate:
        evaluate_models(models=[sweep.finetuned_model], bolden_values=True, include_benchmark=False, ignore_previous_metrics=True)
    
    return sweep


def finetune_on_n_basins(model: TrainedModel, search_space: dict, n=500) -> tuple[list[str], list[Path]]:
    basins = []
    sweeps = []
    generate_run_id = generate_sweep_run_id()
    for _ in range(n):
        # pick basin
        basin, _ = pick_a_basin(model=model)
        basins.append(basin)
        search_space['basin'] = basin

        # finetune a model  
        sweep = find_best_finetuning_params(search_space=search_space, model=model, max_evals=2, evaluate=True)
        sweep_results = sweep.save()
        sweeps.append(sweep_results)
    
    return basins, sweeps