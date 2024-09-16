from experiment.train import train_model
from pathlib import Path
config_path = Path('sota_config.yml')
train_model(config_path)