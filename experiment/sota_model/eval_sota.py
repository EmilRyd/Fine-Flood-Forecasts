# evaluating the SOTA model from HESS paper by Kratzert et al (2020)
from pathlib import Path
import pandas as pd
import pickle as pkl
import glob
import os
from neuralhydrology.nh_run import eval_run

# path to eval data
run_dir = Path(__file__).parent / 'runs' / 'sota_20'
test_path = run_dir / 'test' / 'model_epoch030' / 'test_metrics.csv'

eval_run(run_dir=run_dir, period='test', epoch=30)
# read csv
df = pd.read_csv(test_path)

# average acroos the columns
aves = df.mean()
medians = df.median()
stds = df.std()
print(aves)
print(medians)
print(stds)