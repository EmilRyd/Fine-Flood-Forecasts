# utils used for experiments ,such as loading models and so on
import torch
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.utils.config import Config
from pathlib import Path
from datetime import timedelta
from typing import Union
import os
import pandas as pd
import numpy as np
import re

"""Constants"""
NUM_BASINS = 531

"""Functions"""

def load_cuda_model(config_file: Path, run_dir: Path, epoch: int =30) -> (CudaLSTM, Config):

    """Loads cuda model from config file and run directory, returns tuple of model and config object"""
    
    # instantiate new cudalstm
    cudalstm_config = Config(config_file)
    cuda_lstm = CudaLSTM(cfg=cudalstm_config)

    epoch_string = get_epoch_string(epoch)
    
    model_path = run_dir / f'model_epoch{epoch_string}.pt'
    model_weights = torch.load(str(model_path), map_location='cpu') # load the weights
    cuda_lstm.load_state_dict(model_weights) # set the new mdoel's weights

    return (cuda_lstm, cudalstm_config)

def turn_cuda_into_custom(cuda_lstm, cfg) -> CustomLSTM:

    """Turns CudaLSTM into CustomLSTM for analysis"""
    # load cudalstm weights onto custom lstm
    custom_lstm = CustomLSTM(cfg=cfg)
    custom_lstm.copy_weights(cuda_lstm)
    
    return custom_lstm

def get_epoch_string(epoch: int):
    # load the trained weights into the new model
    epoch_string = str(epoch)
    while len(epoch_string) < 3:
        epoch_string = '0' + epoch_string
    return epoch_string    

def collapse_test_period(cfg: Config) -> Config:
    """Collapse the test period to just one day"""
    cfg_dict = cfg.as_dict()
    cfg_dict['test_end_date'] = cfg.test_start_date + timedelta(days=1)
    new_cfg = Config(cfg_dict)
    return new_cfg

def write_list_to_txt(list_to_write: list, path: str):
    with open(path, 'w') as f:
        f.writelines(f'{item}\n' for item in list_to_write)

def get_cluster_config(base_config: Config, cluster: int, cluster_file: Path) -> Config:
    assert os.path.exists(cluster_file), f'Cluster basin list file {cluster_file} does not exist!'
    cluster_config_dict = base_config._cfg.copy()
    cluster_config_dict['experiment_name'] = base_config.experiment_name + f'cluster{cluster}'
    cluster_config_dict['train_basin_file'] = cluster_file
    cluster_config_dict['test_basin_file'] = cluster_file
    cluster_config = Config(cluster_config_dict)
    

    return cluster_config

def generate_cluster_configs(base_config: Config, cluster_dir: Path) -> list:

    n_clusters = sum(1 for file_ in os.listdir(cluster_dir) if file_.endswith('.txt'))
    cluster_paths = []
    for cluster in range(0, n_clusters):
        cluster_file = cluster_dir / f'{cluster}.txt'
        # only if yml does not already exist
        filename = f'{cluster}.yml'
        cluster_path = cluster_dir / filename
        if not os.path.exists(cluster_path):
            cluster_config = get_cluster_config(base_config=base_config, cluster=cluster, cluster_file=cluster_file)     
            cluster_config.dump_config(folder=cluster_dir, filename=filename)
        cluster_paths.append(cluster_path)
    return cluster_paths

def load_all_caravan_basins():
    # stores all caravan basins in the appropriate txt file
    attr_path = Path(__file__).parent.parent / 'data' / 'Caravan' / 'attributes'
    datasets = ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb', 'hysets', 'lamah']
    all_gauge_ids = []
    for ds in datasets:
        ds_gauge_ids = list(pd.read_csv(attr_path / f'{ds}' / f'attributes_caravan_{ds}.csv')['gauge_id'])
        all_gauge_ids = all_gauge_ids + ds_gauge_ids
    assert len(all_gauge_ids) == len(np.unique(all_gauge_ids)), 'repeating gauge ids'
    basin_file = Path(__file__).parent / 'assets' / 'caravan.txt'
    write_list_to_txt(all_gauge_ids, basin_file)
    return

def get_losses(log_file_path: str):
    

    train_losses = {}
    val_losses = {}

    # Read the log file
    with open(log_file_path, 'r') as f:
        for line in f:
            # Extract training loss using regex
            train_match = re.search(r'avg_total_loss:\s*([\d.]+)', line)
            # Extract validation loss using regex
            val_match = re.search(r'average validation loss:\s*([\d.]+)', line)
            # Extract epochs
            epoch_match = re.search(r'Epoch\s*(\d+)', line)
            
            if train_match:
                assert epoch_match, "no epoch found"
                train_losses[str(epoch_match.group(1))] = (float(train_match.group(1)))
            if val_match:
                assert epoch_match, "no epoch found"
                val_losses[str(epoch_match.group(1))] = (float(val_match.group(1)))
    return train_losses, val_losses


    
"""Classes"""

class TrainedModel:

    def __init__(self, config_file_path_or_experiment_name: Union[Path, str]):
        
        if isinstance(config_file_path_or_experiment_name, Path):
            self.cfg_path = config_file_path_or_experiment_name
        elif isinstance(config_file_path_or_experiment_name, str):
            self.cfg_path = self.get_cfg_path(experiment_name=config_file_path_or_experiment_name)
        else:
            raise ValueError(f'Cannot create a config from input of type {type(config_file_path_or_experiment_name)}.')
        
        self.cfg = Config(self.cfg_path)
        self.config_id = self.cfg.experiment_name
        self.epoch = self.cfg.epochs

        
        self.run_dir = self.cfg_path.parent
        
        epoch_string = get_epoch_string(self.epoch)
        self.metrics_file = (self.run_dir
        / 'test' / f'model_epoch{epoch_string}' / 'test_metrics.csv')

    def get_cfg_path(self, experiment_name: str) -> Path:
        return Path(__file__).parent / 'models' / 'runs' / experiment_name / 'config.yml'

    def get_eval_metrics_file(self, period: str='test') -> Path:
        epoch_string = get_epoch_string(self.epoch)
        return (self.run_dir / period / f'model_epoch{epoch_string}' / f'{period}_metrics.csv') 

    def to_dict(self) -> dict:
        return {
            'best_params': self.best_params,
            'base_model': self.base_model,
            'finetuned_model': self.finetuned_model,
            'basin': self.basin,
            'search_space': self.search_space
        }
class TrainedModelID(StrEnum):
    EMB_20 = 'embedding_experiment_20'
    EMB_10 = 'embedding_experiment_10'
    SOTA_10 = 'sota_10'
    SOTA_20 = 'sota_20'
    SOTA = 'sota'
