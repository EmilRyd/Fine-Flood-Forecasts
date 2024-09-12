"""
Code to cluster basins based on their embeddings
"""
import torch
from pathlib import Path
from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config


config_file = Path('ef_config.yml')

# load trained cudalstm model
run_dir = Path('run dir path')

# instantiate new cudalstm
cudalstm_config = Config(config_file)
cuda_lstm = CudaLSTM(cfg=cudalstm_config)

# load the trained weights into the new model
model_path = run_dir / 'model_epoch016. pt'
model_weights = torch.load(str(model_path), map_location='cuda:0') # load the weights
cuda_lstm.load_state_dict(model_weights) # set the new mdoel's weights

# load cudalstm weights onto custom lstm
custom_lstm = CustomLSTM(cfg=cudalstm_config)
custom_lstm.copy_weights(cuda_lstm)

# sanity check that weights are the same
assert torch.allclose(cuda_lstm.lstm.bias_ih_l0, custom_lstm.cell.b_ih), 'Weights are not the same in CudaLSTM and CustomLSTM'

# todo cluster based on embedding outputs
