
from pathlib import Path
import torch
from neuralhydrology.nh_run import start_run
from experiment.utils import get_trained_model_object

def train_model(config_file_path: Path):
    if torch.cuda.is_available():
        start_run(config_file_path=config_file_path)
    else:
        start_run(config_file_path=config_file_path, gpu=-1)

    # construct TrainedModel object
    trained_model = get_trained_model_object(config_file_path)
    return trained_model


