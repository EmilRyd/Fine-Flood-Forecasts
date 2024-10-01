# redirecting the gradients to the inputs

from pathlib import Path

from experiment.train import train_model

if __name__ == '__main__':
    config_file_path = Path(__file__).parent.parent / 'models' / 'sota_config.yml'
    model = train_model(config_file_path)