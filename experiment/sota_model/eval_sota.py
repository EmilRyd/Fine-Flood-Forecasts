# evaluating the SOTA model from HESS paper by Kratzert et al (2020)
from pathlib import Path
import pandas as pd
import pickle as pkl
import glob
import os
from evaluate import start_evaluation
# path to eval data
test_path = Path(__file__).parent / 'runs' / 'sota' / 'test' / 'model_epoch030' / 'test_metrics.csv'

# read csv
df = pd.read_csv(test_path)

# average acroos the columns
aves = df.mean()
medians = df.median()
stds = df.std()

# get values from all the hess model outputs and compare
hess_path = Path(__file__).parent.parent.parent / 'data' / 'hess_model' / 'cudalstm_all_forcings_seed66401_3004_1400'
file_dirs = glob.glob(str(hess_path))

for path in file_dirs:
    start_evaluation()
    
    


    # average acroos the columns
    aves = df.mean()
    medians = df.median()
    stds = df.std()

    print(aves)

