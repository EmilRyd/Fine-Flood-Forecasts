# Class for trained models, to store their run file path and name
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from experiment.experiment_utils import get_epoch_string

@dataclass
class TrainedModel:
    
    name: str
    config_id: str
    experiment: str
    epoch: int = 30
    _run_dir: Path = None
    _metrics_file: Path = None
    _config_file: Path = None

    @property
    def run_dir(self):
        return Path(__file__).parent / self.experiment / 'runs' / self.config_id
        
    @property
    def metrics_file(self):
        epoch_string = get_epoch_string(self.epoch)
        return (Path(__file__).parent / self.experiment / 'runs' / self.config_id
        / 'test' / f'model_epoch{epoch_string}' / 'test_metrics.csv')

    @property
    def config_file(self):
        return Path(__file__).parent / self.experiment / 'runs' / self.config_id / 'config.yml'
        