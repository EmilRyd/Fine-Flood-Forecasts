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
    fe.run()

if __name__ == '__main__':

    # define args
    # TODO implement this as args here
    parser = argparse.ArgumentParser(description="getting args for the experiment")

    parser.add_argument('config', type=str)

    args = parser.parse_args()
    
    config_file = args.config
    
    # replace the base model path in the config file
    '''finetuning_args = FineTuningArgs.from_yaml(config_file)
    finetuning_args.save(config_file)'''

    finetuner(config_file)
