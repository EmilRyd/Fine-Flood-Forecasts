"""
Code to cluster basins based on their embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from pathlib import Path
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
import os

from experiment.experiment_utils import load_cuda_model, turn_cuda_into_custom, NUM_BASINS, write_list_to_txt
from experiment.trained_models import EMB_MODEL_10

"""
load_model(config_file, run_dir)
load_into_custom_model()
get_embeddings(custom_model)
generate_clusters(embeddings)
return clusters
"""

embeddding_file_path = Path(__file__).parent / 'embeddings.pth'

def get_embeddings(custom_lstm, cfg, run_dir):

    if os.path.exists(embeddding_file_path):
        embeddings = torch.load(embeddding_file_path)
    else:
        # get the statics embedding layer
        custom_lstm.eval()
        s_embedding = custom_lstm.embedding_net.statics_embedding
        
        # load static attributes data
        scaler = load_scaler(run_dir)
        dataset = get_dataset(cfg, is_train=False, period='test', scaler=scaler)

        # calculate embeddings, store them in dict with the basin id as the key
        embeddings = {}
        with torch.no_grad():
            for basin, features in dataset._attributes.items():
                embedded_features = s_embedding(features)
                embeddings[basin] = embedded_features
        
        # store the embeddings
        torch.save(embeddings, embeddding_file_path)
    
    assert len(embeddings.keys()) == NUM_BASINS, f'did not retrive as many basins as embeddings. Expected: {NUM_BASINS}. Received: {len(embeddings.keys())}'

    return embeddings


# cluster the embeddings using K-means algorithm as default
def cluster(embeddings, n_clusters=10, method=KMeans):
    
    method_n = method(n_clusters=n_clusters, random_state=0)
    emb_tensors = torch.stack(list(embeddings.values()))
    method_n.fit(emb_tensors)

    return (method_n)

def investigate_clusters(embeddings, max_clusters=50):
    # see how many clusters are good
    errors = []
    n_clusters = range(1,max_clusters)
    for i in n_clusters:
        fitted_cluster = cluster(embeddings=embeddings, n_clusters=i)
        errors.append(fitted_cluster.inertia_)

    plt.plot(n_clusters, errors)
    plt.show()


# load trained cudalstm model
# TODO figure out how where to pass these arguments
model = EMB_MODEL_10
investigate = True
# get trained cuda model
(cuda_model, cfg) = load_cuda_model(config_file=model.config_file, run_dir=model.run_dir)

# turn it into a custom model
custom_lstm = turn_cuda_into_custom(cuda_model, cfg)

# get the embeddings from the model
embeddings = get_embeddings(custom_lstm, cfg, model.run_dir)

if investigate:
    investigate_clusters(embeddings=embeddings)

# retreive clusters
fitted_cluster = cluster(embeddings)

labels = fitted_cluster.labels_
cluster_dir = f'clustered_basins_kmeans_{len(np.unique(labels))}'

# TODO identify where these dirs should be a nd how they should be named
if not os.path.exists(cluster_dir):
    os.mkdir(cluster_dir)
for label in np.unique(labels):
    basin_cluster = [val for idx, val in enumerate(embeddings.keys()) if (labels[idx] == label)]
    write_list_to_txt(basin_cluster, os.path.join(cluster_dir, f'{label}'))

os.path(cluster_dir)