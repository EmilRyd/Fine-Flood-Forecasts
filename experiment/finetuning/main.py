# function to run automated finetuning on a basin
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params, finetune_model
from hyperopt import hp
from experiment.eval import evaluate_models
from experiment.utils import TrainedModel, TrainedModelID
from IPython.core.display import display, HTML

def param_dict_from_model_output(best_params: dict, basin: str):
    args = {}
    args['basin'] = basin
    args['epochs'] = best_params['epochs']
    args['learning_rate'] = {0: best_params['lr1'], 5: best_params['lr2']}
    args['loss'] = 'NSE'
    args['lstm'] = best_params['lstm']
    return args
    

def main():

    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)
    # pick a basin and get its current nse
    basin, _ = pick_a_basin(model=model)
    
    # define hyperparameter search space
    search_space = {
        'basin': basin,
        'epochs': hp.choice('epochs', [1,2]),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [True, False]),
        'loss': 'NSE'
    }

    # find the best hyperparameters for fine-tuning
    best_params = find_best_finetuning_params(search_space=search_space)

    # add basin back to best params
    args = param_dict_from_model_output(best_params, basin)

    # TODO: save some of this data in a way where you can load it from here
    finetuning_data = finetune_model(args)
    finetuned_model = finetuning_data['model']
    # TODO: Check that validation is happening properly, they are not just reusing old validation data fom previous runs?

    # compare finetuned model and model
    basin_df = evaluate_models(models=[model, finetuned_model], basins=[basin], bolden_values=True, include_benchmark=False)

    # display the comparison
    display(HTML(basin_df.to_html(escape=False)))

    # now compare how the model performs on all basins
    all_df = evaluate_models(models=[model, finetuned_model], bolden_values=True)

    # display the comparison
    display(HTML(all_df.to_html(escape=False)))

    # compare the ratios of training and validation over the training and the finetuning period

if __name__ == '__main__':
    main()