
from pathlib import Path
import torch
from neuralhydrology.nh_run import start_run, start_training
from experiment.utils import TrainedModel
from neuralhydrology.utils.config import Config
import os

def train_model(config_file_path: Path) -> TrainedModel:
    if torch.cuda.is_available():
        start_run(config_file=config_file_path)
    else:
        start_run(config_file=config_file_path, gpu=-1)

    # construct TrainedModel object
    trained_model = TrainedModel(config_file_path_or_experiment_name=config_file_path)

    return trained_model

def train_models(config_paths: list) -> list:
    
    models = []
    for config_path in config_paths:
        config = Config(config_path)
        if not os.path.exists(config.run_dir / config.experiment_name):
            trained_model = train_model(config_path)
        else:
            trained_model = TrainedModel(config_file_path_or_experiment_name=config_path)
        models.append(trained_model)

    return models
