from experiment.utils import TrainedModel, TrainedModelID
import pandas as pd
model = TrainedModel(TrainedModelID.SOTA_20)

df = pd.read_csv(model.metrics_file, dtype={'basin':str})

basin_data = df.loc[df['NSE'] < df['NSE'].median()].sample(n=1)

basin = basin_data.basin.iloc[0]

#with open('/home/admin/Fine-Flood-Forecasts/experiment/models/finetune_basin.txt', 'w') as fp:
#    fp.write(basin)

from neuralhydrology.nh_run import finetune, eval_run
from pathlib import Path

finetune_path = '/home/admin/Fine-Flood-Forecasts/experiment/models/finetune.yml'


# finetune
#finetune(Path(finetune_path))

# evaluate finetuned model
finetune_dir = Path(__file__).parent / 'runs' / 'cudalstm_531_basins_finetuned'
eval_run(finetune_dir, period='test', epoch=10)

# 