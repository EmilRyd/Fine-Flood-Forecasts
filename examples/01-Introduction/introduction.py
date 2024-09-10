import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


import pickle

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run


if __name__ == "__main__":
    if torch.cuda.is_available():
        start_run(config_file=Path('1_basin.yml'))
    else:
        start_run(config_file=Path('1_basin.yml'), gpu=-1)

with open(run_dir / "test" / "model_epoch050" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

print(results.keys())

# extract observations and simulations
qobs = results['camels_03237500']['1D']['xr']['streamflow_obs']
qsim = results['camels_03237500']['1D']['xr']['streamflow_sim']

fig, ax = plt.subplots(figsize=(16,10))
ax.plot(qobs['date'], qobs)
ax.plot(qsim['date'], qsim)
ax.set_ylabel("Discharge (mm/d)")
ax.set_title(f"Test period - NSE {results['camels_03237500']['1D']['NSE']:.3f}")

values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
for key, val in values.items():
    print(f"{key}: {val:.3f}")

plt.close('all')
