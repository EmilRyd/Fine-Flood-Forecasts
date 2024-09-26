# utils for finetuning experiment
from pathlib import Path
import os
import pickle as p

import pandas as pd
from hyperopt import Trials
from tensorboard.backend.event_processing import event_accumulator

from experiment.utils import TrainedModel


# functions
def param_dict_from_model_output(best_params: dict, basin: str):
    args = {}
    args['basin'] = basin
    args['epochs'] = int(best_params['epochs']) + 7
    args['learning_rate'] = {0: float(best_params['lr1']), 5: float(best_params['lr2'])}
    args['loss'] = 'NSE'
    args['lstm'] = best_params['lstm']
    return args

def make_unique(filename):
    counter = 1
    name, extension = os.path.splitext(filename)
    while os.path.exists(filename):
        filename = f"{name}_{counter}{extension}"
        counter += 1
    return filename

def load_pkl(filename: Path):
    with open(filename, 'rb') as f:
        data = p.load(f)
    return data

def get_training_losses(model: TrainedModel) -> dict:
    
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

    def save(self) -> Path:
        results_dir = Path(__file__).parent / 'results'
        unique_filename = make_unique(results_dir / f'{self.base_model.config_id}_{self.basin}.pkl')
        with open(unique_filename, 'wb') as f:
            p.dump(self, f)
        return unique_filename
