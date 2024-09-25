# finetuning experiment code
from pathlib import Path
import pickle as p
from IPython.core.display import display, HTML

from experiment.eval import evaluate_models

from experiment.utils import TrainedModel, TrainedModelID
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params
from hyperopt import hp

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

    # experiment #0, compare for the finetuned basin
    show_performance_comparison(models=[sweep.base_model, sweep.finetuned_model], basins=[sweep.basin])
    
    # experiment #2, compare the ratios of training and validation over the training and the finetuning period    

if __name__ == '__main__':
    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)

    lower = 0.0
    higher = 0.5

    basin, nse = pick_a_basin(model=model, lower=lower, higher=higher)

    # define hyperparameter search space
    search_space = {
        'basin': basin,
        'epochs': 1 + hp.randint('epochs', 1),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [False, True]),
        'loss': 'NSE'
    }
    print(basin)
    print(nse)

    # run the finetuning
    sweep = find_best_finetuning_params(search_space=search_space, model=model, max_evals=1, evaluate=True)
    sweep_results = sweep.save()

    sweep_results = f'results/sota_20_{basin}.pkl'
    perform_experiments(sweep_results=sweep_results)