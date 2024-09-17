# pipeline main function for clustering experiment
from utils import TrainedModel
from pathlib import Path
import os


def main(model: TrainedModel = None, config_file_path: Path = None):
    """
    1. Train a model with an embedding (train or main?)
    2. Evaluate its performance (evaluate)
    3. Generate clusters (clustering)
    4. Train separate models for (train or main?)
    5. Evaluate separate models (evaluate)
    """

    # train a model if one is not provided
    if not model:
        assert config_file_path, 'no file path provided for the config file'
        assert os.path.exists(config_file_path), f'provided file path {config_file_path} does not exist'

        model = train_model(config_file_path)