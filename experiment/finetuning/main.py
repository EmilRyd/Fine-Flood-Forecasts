# function to run automated finetuning on a basin
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params
from hyperopt import hp
from experiment.utils import TrainedModel, TrainedModelID
from experiment.finetuning.experiments import perform_experiments

   
def main(model: TrainedModel, basin: str, search_space: dict):

    # run the finetuning
    sweep = find_best_finetuning_params(search_space=search_space, model=model, max_evals=2)
    sweep_results = sweep.save()

    perform_experiments(sweep_results)

if __name__ == '__main__':

    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)

    lower = 0.0
    higher = 1.0

    basin = None

    if basin is None:
        # pick a basin and get its current nse
        basin, _ = pick_a_basin(model=model, lower=lower, higher=higher)

    # define hyperparameter search space
    search_space = {
        'basin': basin,
        'epochs': hp.choice('epochs', [1,2,3,4]),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [True, False]),
        'loss': 'NSE'
    }

    main(model=model, basin=basin, search_space=search_space)