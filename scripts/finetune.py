#!DOCSTRING: take config file path as an input and then just read the finetuning args from there

# function to run automated finetuning on a basin
from classes.FineTuningArgs import FineTuningArgs
from classes.FineTuningExperiment import FineTuningExperiment

from pathlib import Path
import argparse

def finetuner(config):

    # select base model, start with SOTA Camels model
    cfg_path = Path(f'{config}')

    # define hyperparameter search space
    finetuning_args = FineTuningArgs.from_yaml(cfg_path)
  
    # instantiate finetuning experiment
    fe = FineTuningExperiment(experiment_args=finetuning_args)

    # run finetuning
    fe.run(retrain=True)

if __name__ == '__main__':

    # define args
    parser = argparse.ArgumentParser(description="getting args for the experiment")

    parser.add_argument('config', type=str)

    args = parser.parse_args()
    
    config_file = args.config
    
    finetuner(config_file)
