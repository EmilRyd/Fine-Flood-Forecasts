# utils for finetuning experiment
from pathlib import Path
import os
import json

from experiment.utils import TrainedModel

# functions
def param_dict_from_model_output(best_params: dict, basin: str):
    args = {}
    args['basin'] = basin
    args['epochs'] = int(best_params['epochs'])
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

# classes
class Sweep:

    def __init__(self, best_params: dict, base_model: TrainedModel, finetuned_model: TrainedModel, search_space: dict):

        self.best_params = best_params
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.basin = search_space['basin']
        self.search_space = search_space

    def save(self) -> Path:
        results_dir = Path(__file__).parent / 'results'
        unique_filename = make_unique(results_dir / f'{self.base_model}_{self.basin}.json')
        with open(unique_filename, 'w') as f:
            json.dump(self, f, indent=4)
        return unique_filename