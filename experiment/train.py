"""
Code to train a model using an embedding layer on static attributes on the Caravan dataset
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))
import pickle

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

def train_model(config_path):
    """
    Train model on global Caravan dataset, using an embedding layer on the static attributes
    :return:
    """

    # train model using config file, using cuda-enabled gpu if available
    if torch.cuda.is_available():
        start_run(config_file=config_path)
    else:
        start_run(config_file=config_path, gpu=-1)

if __name__ == '__main__':
    train_model()



