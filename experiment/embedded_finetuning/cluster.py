"""
Code to cluster basins based on their embeddings
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config

"""
load_model(config_file, run_dir)
load_into_custom_model()
get_embeddings(custom_model)
generate_clusters(embeddings)
return clusters
"""


config_file = Path('ef_config.yml')

# load trained cudalstm model
run_dir = Path('runs') / 'embedding_experiment_1109_225229'

def load_model(config_file, run_dir):
    # instantiate new cudalstm
    cudalstm_config = Config(config_file)
    cuda_lstm = CudaLSTM(cfg=cudalstm_config)

    # load the trained weights into the new model
    model_path = run_dir / 'model_epoch016.pt'
    model_weights = torch.load(str(model_path), map_location='cuda:0') # load the weights
    cuda_lstm.load_state_dict(model_weights) # set the new mdoel's weights

    return cudalstm_config, cuda_lstm

def get_embeddings(custom_stm, dataloader):

    embeddings = []
    with torch.no_grad():
        for sample in dataloader:
            # TODO compute the embedding from the static variables, 
            # then associate that with the corresponding basin somehow?
            custom_lstm.embedding_net.statics_embedding(sample['x_s'])
    print(lstm_output[0].keys())


    raise NotImplementedError

def cluster(embeddings):
    
    raise NotImplementedError



cudalstm_config, cuda_lstm = load_model(config_file=config_file, run_dir=run_dir)

# load cudalstm weights onto custom lstm
custom_lstm = CustomLSTM(cfg=cudalstm_config)
custom_lstm.copy_weights(cuda_lstm)
custom_lstm.eval()
print(custom_lstm)

# load the dataset
scaler = load_scaler(run_dir)
dataset = get_dataset(cudalstm_config, is_train=False, period='test', scaler=scaler)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, collate_fn=dataset.collate_fn)

# get embeddings
embeddings = get_embeddings(custom_lstm, dataloader)


# sanity check that weights are the same
assert torch.allclose(cuda_lstm.lstm.bias_ih_l0, custom_lstm.cell.b_ih), 'Weights are not the same in CudaLSTM and CustomLSTM'

# TODO cluster based on embedding outputs

