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

from experiment.utils import load_cuda_model, turn_cuda_into_custom, NUM_BASINS, write_list_to_txt, TrainedModel

"""
load_model(config_file, run_dir)
load_into_custom_model()
get_embeddings(custom_model)
generate_clusters(embeddings)
return clusters
"""

output_dir = Path(__file__).parent / 'outputs'
embeddding_file_path = output_dir / 'embeddings.pth'

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
def cluster(embeddings, n_clusters=1, method=KMeans):
    
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
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def generate_clusters_in_embedding_space(model: TrainedModel, investigate: bool = True, method=KMeans, n_clusters=4) -> Path:
    
    # get trained cuda model
    (cuda_model, cfg) = load_cuda_model(config_file=model.cfg_path, run_dir=model.run_dir)

    # turn it into a custom model
    custom_lstm = turn_cuda_into_custom(cuda_model, cfg)

    # get the embeddings from the model
    embeddings = get_embeddings(custom_lstm, cfg, model.run_dir)

    if investigate:
        investigate_clusters(embeddings=embeddings)

    # retreive clusters
    fitted_cluster = cluster(embeddings, method=method, n_clusters=n_clusters)

    labels = fitted_cluster.labels_

    # write clusters to txt files
    # TODO incorporate the clustering method in the folder naming
    cluster_dir = os.path.join(output_dir, 'clusters', f'{model.config_id}_{len(np.unique(labels))}')

    # store path to clusters to return
    path = Path(cluster_dir)

    if os.path.exists(cluster_dir):
        # assume that if directory exists then clustering has happened
        return path
    os.mkdir(cluster_dir)
    

    for label in np.unique(labels):
        basin_cluster = [val for idx, val in enumerate(embeddings.keys()) if (labels[idx] == label)]
        write_list_to_txt(basin_cluster, os.path.join(cluster_dir, f'{label}.txt'))
    
    

    return path
