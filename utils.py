# utils for finetuning experiment
from __future__ import annotations
from typing import Union

from pathlib import Path
import os
import pickle as p
import random
import re
from datetime import timedelta
from strenum import StrEnum

import pandas as pd
import numpy as np
import torch
from hyperopt import Trials
from huggingface_hub import snapshot_download
import torch

from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import eval_run

# constants
LOSSES = ['NSE', 'MSE', 'RMSE']
SINGLE_BASIN_BATCH_SIZES = [16, 32, 64]
HIDDEN_SIZES = [8, 16, 32]


TEMP_DIR: Path = Path(__file__).parent / 'scripts' / 'temp'
OUTPUT_DIR = Path(__file__).parent / 'output'
MODEL_DIR = Path(__file__).parent / 'pretrained_models'
EPOCH_SWITCH = 20
#SINGLE_BASIN_TRAIN_DIR = Path(os.getenv('DATA')) / 'single_basin_train'

TRAIN_FILE = Path(__file__).parent / 'assets' / 'train.yml'
NUM_BASINS = {'camels_us': 531, 'caravan': 6375}
# functions


def load_cuda_model(config_file: Path, run_dir: Path, epoch: int=30) -> (CudaLSTM, Config):

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

def read_txt_to_list(file_path) -> list:
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    return lines

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

# hacky solution to get all the caravan basins with 
def load_all_caravan_basins_with_train_data():
    basins = get_all_basins('caravan')
    file_path = Path(__file__).parent / 'assets' / 'caravan_nodata.txt'
    no_data_basins = read_txt_to_list(file_path)
    basins_set = set(basins)
    no_data_set = set(no_data_basins)

    basins_with_data = basins_set - no_data_set
    file_path = Path(__file__).parent / 'assets' / 'caravan_with_data.txt'
    write_list_to_txt(list(basins_with_data), file_path)
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

def get_all_basins(dataset):
    # return all the basins in a dataset as a list, based on txt file where they are already generated
    assets_dir = Path(__file__).parent / 'assets'
    file_path = assets_dir / (dataset + '.txt')
    assert os.path.exists(file_path), f'file path {file_path} with all basins does not exist'
    return read_txt_to_list(file_path)
    

def pick_basins(n_basins, config_file_path) -> Path:
    """pick n basins and update the config file to point to them"""
    cfg = Config(config_file_path)
    # read all the basins from the dataset
    all_basins = get_all_basins(cfg.dataset.lower())

    # pick n elements randomly from list
    picked_basins = random.sample(all_basins, n_basins)

    #TODO: maybe perform some safety check here seeing if this dataset will work (no zero std and so on)

    # write list to txt file
    n_basin_file = Path(__file__).parent / 'assets' / (cfg.dataset.lower() + '_' + str(n_basins) + '.txt')
    write_list_to_txt(picked_basins, n_basin_file)

    # update config
    cfg._cfg['train_basin_file'] = n_basin_file
    cfg._cfg['validation_basin_file'] = n_basin_file
    cfg._cfg['test_basin_file'] = n_basin_file


    # TODO: put the dumping of yamls in a proper util func (and make it safe!)
    # write back to the YAML file
    new_filename = f'{cfg.dataset.lower()}_{n_basins}.yml'
    cfg.dump_config(folder=config_file_path.parent, filename=new_filename)
    
    new_file_path = config_file_path.parent / new_filename

    return new_file_path

def get_basin_file(basin: str) -> Path:
    return Path(__file__).parent / 'assets' / 'caravan_basin_files' / f'{basin}.txt'

def param_dict_from_model_output(best_params: dict, basin: str):
    args = {}
    args['basin'] = basin
    args['epochs'] = int(best_params['epochs'])
    args['learning_rate'] = {0: float(best_params['lr1']), EPOCH_SWITCH: float(best_params['lr2'])}
    args['loss'] = LOSSES[best_params['loss']]
    #args['lstm'] = best_params['lstm']
    args['batch_size'] = SINGLE_BASIN_BATCH_SIZES[best_params['batch_size']]
    args['hidden_size'] = HIDDEN_SIZES[best_params['hidden_size']]
    return args

def make_unique(name):
    counter = 1
    base_name, extension = os.path.splitext(name)
    while os.path.exists(name):
        name = Path(f"{base_name}_{counter}{extension}")
        counter += 1
    return name

def generate_sweep_run_directory(experiment_id: str, output_dir: Path):
    results_dir = output_dir / 'sweeps'
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    dirname = results_dir / experiment_id
    u_name = make_unique(name=dirname)
    os.mkdir(u_name)
    return u_name



def load_pkl(filename: Path):
    with open(filename, 'rb') as f:
        data = p.load(f)
    return data



def generate_basin_txt_files(basins: list):
    basin_dir = Path(__file__).parent / 'assets' / 'caravan_basin_files'
    if not os.path.exists(basin_dir):
        os.mkdir(basin_dir)
    for basin in basins:
        with open(basin_dir / f'{basin}.txt', 'w') as f:
            f.write(basin)


def filter_outliers(value):
    # Step 1: Calculate Q1, Q3, and IQR
    Q1 = np.percentile(value, 25)
    Q3 = np.percentile(value, 75)
    IQR = Q3 - Q1

    # Step 2: Define the outlier thresholds
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR

    # Step 3: Filter out the outliers
    no_outliers_data = [x for x in value if lower_bound <= x <= upper_bound]
    return no_outliers_data

def load_hf_model():

   
    # prepare directory for download
    os.makedirs(MODEL_DIR, exist_ok=True)

    # download from hugging face
    print('Loading model from Hugging Face...')
    snapshot_download(
        repo_id='EmilRyd/caravan_model', 
        allow_patterns='caravan_base/*',
        local_dir=MODEL_DIR,
        repo_type='model'
    )    
    # turn into TrainedModel
    base_model = TrainedModel(MODEL_DIR / 'caravan_base' / 'config.yml')
    print('Model loaded and saved.')


    return base_model

# classes
'''class Sweep:

    def __init__(self, best_params: dict, base_model: TrainedModel, finetuned_model: TrainedModel, search_space: dict, max_evals: int, trials: Trials):

        self.best_params = best_params
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.basin = search_space['basin']
        self.search_space = search_space
        self.max_evals = max_evals
        self.trials = trials

    def save(self, run_dir: Path) -> Path:
        filename = run_dir / f'{self.basin}.pkl'
        with open(filename, 'wb') as f:
            p.dump(self, f)
        return Path(filename)
    '''
class TrainingExperimentResults:

    def __init__(self, best_params: dict, final_model: TrainedModel, search_space: dict, max_evals: int, trials: Trials, basin: str, base_model=None):

        self.best_params = best_params
        self.base_model = base_model
        self.final_model = final_model
        self.basin = basin
        self.search_space = search_space
        self.max_evals = max_evals
        self.trials = trials

    def save(self, run_dir: Path) -> Path:
        filename = run_dir / f'{self.basin}.pkl'
        with open(filename, 'wb') as f:
            p.dump(self, f)
        return Path(filename)

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
    
    def get_eval_metrics(self, basins: list = None, period: str='test') -> pd.DataFrame:

        metrics_file = self.get_eval_metrics_file(period=period)
        if not os.path.exists(metrics_file):
            print(f'Metrics file {metrics_file} does not exist, running evaluation now...')
            if torch.cuda.is_available():
                eval_run(self.run_dir, period=period, epoch=self.epoch)
            else:
                eval_run(self.run_dir, period=period, epoch=self.epoch, gpu=-1)
        
        df = pd.read_csv(metrics_file)


        if basins:
            assert sum(df.basin.isin(basins)) == len(basins), 'Basin column not in metrics file'
            return df[df['basin'].isin(basins)]
        else:
            return df
         

    def to_dict(self) -> dict:
        return {
            'best_params': self.best_params,
            'base_model': self.base_model,
            'finetuned_model': self.finetuned_model,
            'basin': self.basin,
            'search_space': self.search_space
        }
