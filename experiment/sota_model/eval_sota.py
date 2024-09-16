# evaluating the SOTA model from HESS paper by Kratzert et al (2020)
from pathlib import Path
import pandas as pd

# path to eval data
test_path = Path(__file__).parent / 'runs' / 'embedding_experiment_1609_103917' / 'test' / 'model_epoch030' / 'test_metrics.csv'

# read csv
df = pd.read_csv(test_path)

# average acroos the columns
aves = df.mean()
medians = df.median()
stds = df.std()

print(aves)
print(stds)
print(medians)