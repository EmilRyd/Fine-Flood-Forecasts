# pipeline main function for clustering experiment
import os
from pathlib import Path


from experiment.utils import TrainedModel, generate_cluster_configs, TrainedModelID
from experiment.eval import evaluate_models
from experiment.clustering import generate_clusters_in_embedding_space
from experiment.train import train_model, train_models

def main(model: TrainedModel = None, config_file_path: Path = None):
    """
    1. Train a model with an embedding (train)
    2. Evaluate its performance (evaluate)
    3. Generate clusters (clustering)
    4. Train separate models for (train)
    5. Evaluate separate models (evaluate)
    """

    # train a model if one is not provided
    if model is None:
        assert config_file_path, 'no file path provided for the config file'
        assert os.path.exists(config_file_path), f'provided file path {config_file_path} does not exist'

        model = train_model(config_file_path)
    
    # evaluate model (together with other models if you wish)
    models = [model]
    evaluate_models(models)

    # generate clusters
    cluster_dir = generate_clusters_in_embedding_space(model=model)

    # generate new config files
    cluster_config_paths = generate_cluster_configs(base_config=model.cfg, cluster_dir=cluster_dir)

    # train the new models
    clustered_models = train_models(cluster_config_paths)
    
    all_models = clustered_models + models
    
    # evaluate the separate models
    evaluate_models(all_models)

if __name__ == '__main__':
    # perform clustering experiment on a given mode
    config_file_path = Path(__file__).parent / 'models' / 'emb_caravan_config.yml'
    main(config_file_path=config_file_path)