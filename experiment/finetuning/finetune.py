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
from experiment.finetuning.utils import Sweep, cfg_from_args, generate_sweep_run_directory, param_dict_from_model_output
import logging
from hyperopt import fmin, Trials, tpe, STATUS_OK
from datetime import datetime
from neuralhydrology.utils.config import Config


# pick a basin
def pick_a_basin(model: TrainedModel, lower: float = None, higher: float = None) -> str:
    df = pd.read_csv(model.get_eval_metrics_file(period='validation'), dtype={'basin':str})
    c_time = int(datetime.now().timestamp())
    if lower is None or higher is None:
        basin_data = df.sample(n=1, random_state=c_time)
    else:
        basin_data = df.loc[(df['NSE'] <= higher) & (df['NSE'] >= lower)].sample(n=1, random_state=c_time)

    basin = basin_data.basin.iloc[0]
    return basin

def update_files(model: TrainedModel, basin: str, yml_file_path=os.path.join('assets', 'finetune.yml')):
    
    # Add the path to the pre-trained model to the finetune config
    with open(yml_file_path, "a") as fp:
        fp.write(f"\nbase_run_dir: {model.run_dir.absolute()}")

    # Load the existing YAML data
    with open(yml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    data['experiment_name'] = f'basin_{basin}'  # Example modification

    basin_file_path = os.path.join('assets', 'basin_files', f'{basin}.txt')
    data['train_basin_file'] = basin_file_path
    data['validation_basin_file'] = basin_file_path

    # Write back to the YAML file
    with open(yml_file_path, 'w') as f:
        yaml.dump(data, f)  

    
    # Create a basin file with the basin we selected above
    with open(basin_file_path, 'w') as fp:
        fp.write(basin) 
def finetune_model_from_cfg(data: dict, basin: str):
    # finetune using temporary yaml file
    assets_dir = Path(__file__).parent / 'assets'
    #
    with tempfile.NamedTemporaryFile(delete=True, dir=assets_dir, suffix='.yml', mode='w') as f:
        yaml.dump(data, f)  

        finetuned_model = finetune_model(assets_dir / f.name)

        v_df = evaluate_models([finetuned_model], basins=[basin], include_benchmark=False, period='validation', ignore_previous_metrics=True)
    
    # return negative validation score
    return {'loss': -float(v_df.iloc[0][f'basin_{basin}']), 'status': STATUS_OK, 'model': finetuned_model}

def finetune_model_from_args(args):
    basin = args['basin']
    data = cfg_from_args(args)
    score = finetune_model_from_cfg(data=data, basin=basin)
    return score
    
def finetune_model(base_config_file: Path) -> TrainedModel:
    cfg = Config(base_config_file)
    finetune(base_config_file)
    run_dir = Path(os.getcwd()) / 'runs' / f'{cfg.experiment_name}'
    config_file_path = run_dir / 'config.yml'
    finetuned_model = TrainedModel(config_file_path_or_experiment_name=config_file_path)
    return finetuned_model


def find_best_finetuning_params(search_space: dict, model: TrainedModel, max_evals=10, evaluate=True) -> Sweep:
   
    trials = Trials()
    best_params = fmin(finetune_model_from_args, space=search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # add basin back to best params    
    # add basin back to best params

    # run best model to get that fresh validation data
    best_args = param_dict_from_model_output(best_params, search_space['basin'])
    finetuning_data = finetune_model_from_args(best_args)
    finetuned_model = finetuning_data['model']

    # store model, finetuned model, and best_params
    sweep = Sweep(best_params=best_params, base_model=model, 
                  finetuned_model=finetuned_model, search_space=search_space,
                    max_evals=max_evals, trials=trials)

    # perform evaluation on all basins
    if evaluate:
        evaluate_models(models=[sweep.finetuned_model], bolden_values=True, include_benchmark=False, ignore_previous_metrics=True)
    
    return sweep


def finetune_on_n_basins(model: TrainedModel, search_space: dict, n=500, max_evals=50) -> tuple[list[str], list[Path], Path]:
    basins = []
    sweeps = []
    run_dir = generate_sweep_run_directory(base_model_id=model.config_id)
    
    for _ in range(n):
        # pick basin
        basin = pick_a_basin(model=model)
        while basin in basins:
            basin = pick_a_basin(model=model)
        update_files(model=model, basin=basin)

        basins.append(basin)
        search_space['basin'] = basin

        # finetune a model  
        sweep = find_best_finetuning_params(search_space=search_space, model=model, max_evals=max_evals, evaluate=True)
        sweep_results = sweep.save(run_dir=run_dir)
        sweeps.append(sweep_results)
    
    return basins, sweeps, run_dir