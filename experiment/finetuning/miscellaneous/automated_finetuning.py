# automate the finetuning across all basins
import yaml
from pathlib import Path
import os

import pandas as pd

from neuralhydrology.nh_run import finetune
from experiment.eval import evaluate_models
from experiment.utils import TrainedModel, TrainedModelID


def setup_configs(model: TrainedModel, basin: str):
    file_path = "finetune.yml"

    with open(file_path, "a") as fp:
        fp.write(f"\nbase_run_dir: {model.run_dir.absolute()}")

    # Load the existing YAML data
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    data['experiment_name'] = f'basin_{basin}'  # Example modification
    data['train_basin_file'] = f'assets/finetune_basin_{basin}.txt'
    data['test_basin_file'] = f'assets/finetune_basin_{basin}.txt'
    data['validation_basin_file'] = f'assets/finetune_basin_{basin}.txt'
    # Write back to the YAML file
    with open(file_path, 'w') as f:
        yaml.dump(data, f)   
    
    # Create a basin file with the basin we selected above
    dir_path = os.path.join('assets', f'{model.config_id}')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with open(os.path.join(dir_path, f'finetune_basin_{basin}.txt'), 'w') as fp:
        fp.write(basin)
    


def finetune_model_for_basin(model: TrainedModel, basin: str):
    # finetunes the given model for the given basin, returns the model and the metrics
    # first edit the new yaml and txt file
    setup_configs(model, basin)

    # finetune
    finetune(Path("finetune.yml"))

    # get the finetuned model
    run_dir = Path(os.path.abspath('')) / 'runs' / f'basin_{basin}'
    config_file_path = run_dir / 'config.yml'
    finetuned_model = TrainedModel(config_file_path_or_experiment_name=config_file_path)

    # evaluate the model 
    metrics = evaluate_models([finetuned_model], basins=[basin], include_benchmark=False, period='test')

    return (finetuned_model, metrics)


    


if __name__ == '__main__':
    base_model = TrainedModel(TrainedModelID.SOTA_20)

    # iterate over all basins and perform finetuning on them
    # metrics
    df = pd.read_csv(base_model.metrics_file, dtype={'basin':str})

    # basins
    basins = df.basin

    load_fm_path = os.path.join('assets', f'{base_model.config_id}', 'finetuned_metrics.csv')

    if os.path.exists(load_fm_path):
        finetuned_metrics = pd.read_csv(load_fm_path)
    else:
        finetuned_mettric = pd.DataFrame()
    
    # iterate over basins
    for idx, basin in enumerate(basins):
        if basin not in finetuned_metrics.columns:
            finetuned_model, metrics = finetune_model_for_basin(base_model, basin)

            if finetuned_metrics.empty:
                metrics = metrics.rename(columns={f'basin_{basin}':f'{basin}'})
                finetuned_metrics = metrics
            else:
                finetuned_metrics[basin] = metrics[f'basin_{basin}']
            if idx % 10 == 0:
                finetuned_metrics.to_csv(os.path.join('assets', f'{base_model.config_id}', 'finetuned_metrics.csv'), index=False)
        

    finetuned_metrics.to_csv(os.path.join('assets', f'{base_model.config_id}', 'finetuned_metrics.csv'), index=False)

    # write yaml file into the finetuning assets directory  

    file_path = "finetune.yml"

    # Load the existing YAML data
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Write back to the YAML file
    with open(os.path.join('assets', f'{base_model.config_id}', 'finetune.yml'), 'w') as f:
        yaml.dump(data, f)   