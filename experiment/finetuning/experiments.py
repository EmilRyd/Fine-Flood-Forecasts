# finetuning experiment code
from pathlib import Path
import json
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

    with open(sweep_results, 'r') as f:
        sweep = json.load(f)
    
    # experiment #1, compare across all basins
    show_performance_comparison(models=[sweep.model, sweep.finetuned_model])

    # experiment #0, compare for the finetuned basin
    show_performance_comparison(models=[sweep.model, sweep.finetuned_model], basins=[sweep.basin])
    
