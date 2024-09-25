# function to run automated finetuning on a basin
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params, finetune_model
from hyperopt import hp
from experiment.eval import evaluate_models
from experiment.utils import TrainedModel, TrainedModelID
from IPython.core.display import display, HTML
from pathlib import Path


def show_performance_comparison(models: list, basins: list = []):
    if len(basins) == 0:
        # compare how the model performs on all basins
        df = evaluate_models(models=models, bolden_values=True, include_benchmark=False)
    else:
        # compare finetuned model and model's performance on the selected basin
        df = evaluate_models(models=models, basins=basins, bolden_values=True, include_benchmark=False)

    # display the comparison
    display(HTML(df.to_html(escape=False)))

def run_experiment_from_file(run_file: Path):

    best_params, model, finetuned_model, basin = load_sweep_from_file(run_file)

    # experiment #1, compare across all basins
    show_performance_comparison(models=[model, finetuned_model])

    # experiment #0, compare for the finetuned basin
    show_performance_comparison(models=[model, finetuned_model], basins=[basin])

   
def main(basin: str = None, lower: float = 0.0, higher: float = 1.0):

    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)

    if basin is None:
        # pick a basin and get its current nse
        basin, _ = pick_a_basin(model=model, lower=lower,)
    
    # define hyperparameter search space
    search_space = {
        'basin': basin,
        'epochs': hp.choice('epochs', [1,2,3,4]),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [True, False]),
        'loss': 'NSE'
    }
    
    # run the finetuning
    sweep = find_best_finetuning_params(search_space=search_space, model=model)
    sweep_results = sweep.save()

    show_performance_comparison(sweep_results)

    # experiment #2, compare the ratios of training and validation over the training and the finetuning period
    

if __name__ == '__main__':

    main()