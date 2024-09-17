from experiment.trainedmodel import TrainedModel

# all models
EMB_MODEL_20 = TrainedModel(name='emb_model_20', config_id='embedding_experiment_20', experiment='embedding_model')
EMB_MODEL_10 = TrainedModel(name='emb_model_10', config_id='embedding_experiment_10', experiment='embedding_model')
SOTA_MODEL_10 = TrainedModel(name='sota_model_10', config_id='sota_10', experiment='sota_model')
SOTA_MODEL_20 = TrainedModel(name='sota_model_20', config_id='sota_20', experiment='sota_model')

ALL_MODELS = [EMB_MODEL_20, EMB_MODEL_10, SOTA_MODEL_10, SOTA_MODEL_20]
