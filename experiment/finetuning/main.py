# function to run automated finetuning on a basin
from experiment.finetuning import pick_a_basin, find_best_finetuning_parameters
from hyperopt import hp
from experiment.eval import evaluate_models
from experiment.utils import TrainedModel, TrainedModelID
from IPython.core.display import display, HTML

def main():

    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)
    # pick a basin and get its current nse
    basin, _ = pick_a_basin()
    
    # define hyperparameter search space
    search_space = {
        'epochs': hp.choice('epochs', [1,2,3,4,5,6,7,8,9,10]),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [True, False]),
        'loss': 'NSE'
    }

    # find the best hyperparameters for fine
    best_params = find_best_finetuning_parameters(basin=basin, search_space=search_space)

    # train a model using the best params
    finetuned_model = best_params['model']

    # TODO: save some of this data in a way where you can load it from here

    # compare finetuned model and model
    basin_df = evaluate_models(models=[model, finetuned_model], basins=[basin], bolden_values=True)

    # display the comparison
    display(HTML(basin_df.to_html(escape=False)))

    # now compare how the model performs on all basins
    all_df = evaluate_models(models=[model, finetuned_model], bolden_values=True)

    # display the comparison
    display(HTML(all_df.to_html(escape=False)))

    # compare the ratios of training and validation over the training and the finetuning period

