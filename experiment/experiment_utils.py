# utils used for experiments ,such as loading models and so on
import torch
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.utils.config import Config
from pathlib import Path
from datetime import timedelta

NUM_BASINS = 531

def load_cuda_model(config_file, run_dir, epoch=30) -> (CudaLSTM, Config):

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
    