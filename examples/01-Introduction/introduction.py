import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


import pickle

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

if torch.cuda.is_available():
    start_run(config_file=Path('1_basin.yml'))
else:
    start_run(config_file=Path('1_basin.yml'), gpu=-1)

