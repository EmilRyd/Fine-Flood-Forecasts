"""
Code to cluster basins based on their embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from pathlib import Path
from neuralhydrology.datasetzoo import get_dataset, camelsus
from neuralhydrology.datautils.utils import load_scaler

from neuralhydrology.nh_run import start_run
from experiment.exp_utils import load_cuda_model, turn_cuda_into_custom


"""
load_model(config_file, run_dir)
load_into_custom_model()
get_embeddings(custom_model)
generate_clusters(embeddings)
return clusters
"""

def get_embeddings(custom_stm, cfg, run_dir):
    # get the statics embedding layer
    custom_lstm.eval()
    s_embedding = custom_lstm.embedding_net.statics_embedding
    
    # load static attributes data
    scaler = load_scaler(run_dir)
    dataset = get_dataset(cfg, is_train=False, period='test', scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    # calculate embeddings, store them in dict with the basin id as the key
    embeddings = {}
    with torch.no_grad():
        for basin, features in dataset._attributes.items():
            embedded_features = s_embedding(features)
            embeddings[basin] = embedded_features
    
    return embeddings






    # get the basins
    print('helo')


config_file = Path('ef_config.yml')

# load trained cudalstm model
# TODO figure out how where to pass these arguments
run_dir = Path(__file__).parent / 'runs' / 'embedding_experiment_1609_112205'

# get trained cuda model
(cuda_model, cfg) = load_cuda_model(config_file=config_file, run_dir=run_dir, epoch=1)

# turn it into a custom model
custom_lstm = turn_cuda_into_custom(cuda_model, cfg)

# get the embeddings from the model
embeddings = get_embeddings(custom_lstm, cfg, run_dir)

# cluster the embeddings using K-means algorithm

# see how many clusters are good
errors = []
max_clusters = 20
n_clusters = range(1,max_clusters):
for i in n_clusters:
    kmeans = KMeans(n_clusters=i, random_state=0)
    emb_tensors = torch.stack(list(embeddings.values()))
    kmeans.fit(emb_tensors)

    errors.append(kmeans.inertia_)

plt.plot(n_clusters, errors)
plt.show()




def cluster(embeddings):
    
    raise NotImplementedError







# sanity check that weights are the same
assert torch.allclose(cuda_lstm.lstm.bias_ih_l0, custom_lstm.cell.b_ih), 'Weights are not the same in CudaLSTM and CustomLSTM'

# TODO cluster based on embedding outputs

