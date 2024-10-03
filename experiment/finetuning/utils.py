# utils for finetuning experiment
from pathlib import Path
import os
import pickle as p

import pandas as pd
from hyperopt import Trials
from sympy import Union
from tensorboard.backend.event_processing import event_accumulator
import yaml
from datetime import datetime

from experiment.utils import TrainedModel

# constants
LOSSES = ['NSE', 'MSE', 'RMSE']

# functions
def param_dict_from_model_output(best_params: dict, basin: str):
    args = {}
    args['basin'] = basin
    args['epochs'] = int(best_params['epochs'])
    args['learning_rate'] = {0: float(best_params['lr1']), 30: float(best_params['lr2'])}
    args['loss'] = LOSSES[best_params['loss']]
    args['lstm'] = best_params['lstm']
    return args

def cfg_from_args(args):

    # sanity check on args
    # get base cfg
    yml_file_path = Path(__file__).parent / 'assets' / 'finetune.yml'

   # Load the existing YAML data
    with open(yml_file_path, 'r') as f:
        data = yaml.safe_load(f)

    # set dict parameters based on config dictionary passed to function
    modules = ['head'] 
    if args['lstm']:
        modules.append('lstm')
    data['epochs'] = int(args['epochs'])
    data['learning_rate'] = args['learning_rate']
    data['loss'] = args['loss']
    data['finetune_modules'] = modules
    
    return data

def make_unique(name):
    counter = 1
    base_name, extension = os.path.splitext(name)
    while os.path.exists(name):
        name = Path(f"{base_name}_{counter}{extension}")
        counter += 1
    return name

def generate_sweep_run_directory(base_model_id: str):
    results_dir = Path(__file__).parent / 'sweeps'
    dirname = results_dir / base_model_id
    u_name = make_unique(name=dirname)
    os.mkdir(u_name)
    return u_name



def load_pkl(filename: Path):
    with open(filename, 'rb') as f:
        data = p.load(f)
    return data

def get_training_losses(model: TrainedModel) -> tuple:
    
    # returns training and validation losses for a trained model
    assert model.cfg.log_tensorboard, f'Tensorboard logging set to {model.cfg.log_tensorboard} for this basin, such behavior not supported yet'

    log_dir = str(model.run_dir)

    # load the tensor board data
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance=
                                            {event_accumulator.SCALARS: 0})
    ea.Reload()

    # access the losses
    t_loss_data = ea.Scalars('train/avg_loss')
    v_loss_data = ea.Scalars('valid/avg_loss')

    train_df = pd.DataFrame(t_loss_data)
    val_df = pd.DataFrame(v_loss_data)

    return train_df, val_df


# classes
class Sweep:

    def __init__(self, best_params: dict, base_model: TrainedModel, finetuned_model: TrainedModel, search_space: dict, max_evals: int, trials: Trials):

        self.best_params = best_params
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.basin = search_space['basin']
        self.search_space = search_space
        self.max_evals = max_evals
        self.trials = trials

    def save(self, run_dir: Path) -> Path:
        unique_filename = make_unique(run_dir / f'{self.basin}.pkl')
        with open(unique_filename, 'wb') as f:
            p.dump(self, f)
        return Path(unique_filename)
