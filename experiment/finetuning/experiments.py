# finetuning experiment code
from pathlib import Path
import pickle as p
from IPython.core.display import display, HTML

from experiment.eval import evaluate_models



def show_performance_comparison(models: list, basins: list = []):
    if len(basins) == 0:
        # compare how the model performs on all basins
        df = evaluate_models(models=models, bolden_values=True, include_benchmark=False)
    else:
        # compare finetuned model and model's performance on the selected basin
        df = evaluate_models(models=models, basins=basins, bolden_values=True, include_benchmark=False)

    # display the comparison
    display(HTML(df.to_html(escape=False)))

def perform_experiments(sweep_results: Path):

    with open(sweep_results, 'rb') as f:
        sweep = p.load(f)
    
    # experiment #1, compare across all basins
    show_performance_comparison(models=[sweep.base_model, sweep.finetuned_model])

    #maybe comapre the validatio loss improvement as welL, for sanity
    

    # experiment #0, compare for the finetuned basin
    show_performance_comparison(models=[sweep.base_model, sweep.finetuned_model], basins=[sweep.basin])
    
    # experiment #2, compare the ratios of training and validation over the training and the finetuning period    

from experiment.utils import TrainedModel, TrainedModelID
from hyperopt import hp
import os
from experiment.finetuning.utils import load_pkl

# reading resluts and performing experiments on them

if __name__ == '__main__':
    results_dir = Path(__file__).parent / 'results'
    for filename in os.listdir(results_dir):
        sweep_results = os.path.join(results_dir, filename)
        if os.path.isfile(sweep_results):
            perform_experiments(sweep_results=sweep_results)

