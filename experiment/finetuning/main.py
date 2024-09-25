# function to run automated finetuning on a basin
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params, finetune_model
from hyperopt import hp
from experiment.eval import evaluate_models
from experiment.utils import TrainedModel, TrainedModelID
from IPython.core.display import display, HTML

def param_dict_from_model_output(best_params: dict, basin: str):
    args = {}
    args['basin'] = basin
    args['epochs'] = int(best_params['epochs'])
    args['learning_rate'] = {0: float(best_params['lr1']), 5: float(best_params['lr2'])}
    args['loss'] = 'NSE'
    args['lstm'] = best_params['lstm']
    return args
    
def show_performance_comparison(models: list, basins: list = []):
    if len(basins) == 0:
        # compare how the model performs on all basins
        df = evaluate_models(models=models, bolden_values=True, include_benchmark=False)
    else:
        # compare finetuned model and model's performance on the selected basin
        df = evaluate_models(models=models, basins=basins, bolden_values=True, include_benchmark=False)

    # display the comparison
    display(HTML(df.to_html(escape=False)))
   
def main(basin: str = None):

    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)

    if basin is None:
        # pick a basin and get its current nse
        basin, _ = pick_a_basin(model=model)
    
    # define hyperparameter search space
    search_space = {
        'basin': basin,
        'epochs': hp.choice('epochs', [1,2,3,4]),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [True, False]),
        'loss': 'NSE'
    }

    # find the best hyperparameters for fine-tuning
    best_params = find_best_finetuning_params(search_space=search_space, max_evals=1)

    # add basin back to best params
    args = param_dict_from_model_output(best_params, basin)

    finetuning_data = finetune_model(args)
    finetuned_model = finetuning_data['model']

    # TODO: order is currently necessary so that vlaidation does happen across all basins. fi to avoid doing this hack.

    # experiment #1, compare across all basins
    show_performance_comparison(models=[model, finetuned_model])

    # experiment #0, compare for the finetuned basin
    show_performance_comparison(models=[model, finetuned_model], basins=[basin])

    # experiment #2, compare the ratios of training and validation over the training and the finetuning period
    

if __name__ == '__main__':

    main()