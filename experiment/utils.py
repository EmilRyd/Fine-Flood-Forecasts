# utils used for experiments ,such as loading models and so on
import torch
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.utils.config import Config
from pathlib import Path
from datetime import timedelta
from typing import Union
import os
from strenum import StrEnum
import numpy as np
import pickle as p
import sys
import dill as d
import pandas as pd
from hyperopt import Trials
from typing import Union
from tensorboard.backend.event_processing import event_accumulator
import yaml

"""Constants"""
NUM_BASINS = 531
LOSSES = ['NSE', 'MSE', 'RMSE']

"""Functions"""

def load_camels_basins() -> list:
    basin_file = Path(__file__).parent / 'assets' / '531_basins.txt'
    return read_txt_to_list(basin_file)

def read_txt_to_list(txt_file: Path) -> list:
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f]

def load_cuda_model(config_file: Path, run_dir: Path, epoch: int =30) -> tuple[CudaLSTM, Config]:

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

def make_unique(name):
    counter = 1
    base_name, extension = os.path.splitext(name)
    while os.path.exists(name):
        name = Path(f"{base_name}_{counter}{extension}")
        counter += 1
    return name

def generate_run_directory(base_model_id: str, type_of_run: str):
    results_dir = Path(os.getcwd()) / type_of_run
    dirname = results_dir / base_model_id
    u_name = make_unique(name=dirname)
    os.mkdir(u_name)
    return u_name



def load_pkl(filename: Path):
    with open(filename, 'rb') as f:
        data = CustomUnpickler(f).load()
    #del sys['experiment.finetuning.utils']
    return data    

"""Classes"""

class TrainedModel:

    def __init__(self, config_file_path_or_experiment_name: Union[Path, str]):
        
        if isinstance(config_file_path_or_experiment_name, Path):
            self.cfg_path = config_file_path_or_experiment_name
        elif isinstance(config_file_path_or_experiment_name, str):
            self.cfg_path = self.get_cfg_path(experiment_name=config_file_path_or_experiment_name)
        else:
            raise ValueError(f'Cannot create a TrainedModel from input of type {type(config_file_path_or_experiment_name)}.')
        
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
    
    def get_training_losses(self) -> tuple:
        # returns training and validation losses for a trained model
        assert self.cfg.log_tensorboard, f'Tensorboard logging set to {self.cfg.log_tensorboard} for this basin, such behavior not supported yet'

        log_dir = str(self.run_dir)

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

class TrainedModelID(StrEnum):
    EMB_20 = 'embedding_experiment_20'
    EMB_10 = 'embedding_experiment_10'
    SOTA_10 = 'sota_10'
    SOTA_20 = 'sota_20'
    SOTA = 'sota'
    ATTRIBUTE_TUNING = 'attribute_tuning'

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

class CustomUnpickler(p.Unpickler):
    def find_class(self, module, name):
        if module == 'experiment.finetuning.utils':
            module = 'experiment.utils'
        return super().find_class(module, name)