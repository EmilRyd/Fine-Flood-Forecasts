# function to run automated finetuning on a basin
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params
from hyperopt import hp
from experiment.utils import TrainedModel, TrainedModelID
from experiment.finetuning.experiments import perform_experiments

if __name__ == '__main__':

    # actual experiment (I think)

    # pick 3 basins

    # for each basin, do the following
        # finetune a model using some set search_space (how do you make sure you finetune a model really well? that should likely be done through a notebook, or should it be done manually?)

        # print out the test set comparison for that basin
        # print out the test set comparison for across all basins
        # plot the validation and training loss for the base model (already stored in output.log), and the validation and training loss for the finetuning model (already stored in output.log)

    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)

    mean = 0.79 # taken raw from the sota model performance
    threshold = 0.05

    lowers = [0,mean-threshold, mean+threshold]
    highers = [mean-threshold, mean+threshold, 1]

    # define hyperparameter search space
    search_space = {
        'epochs': 1 + hp.randint('epochs', 5),
        'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
        'lstm': hp.choice('lstm', [False, True]),
        'loss': 'NSE'
    }


    basins = []
    for idx, lower in enumerate(lowers):
        # pick basin
        basin, _ = pick_a_basin(model=model, lower=lower, higher=highers[idx])

        search_space['basin'] = basin

        # finetune a model  
        sweep = find_best_finetuning_params(search_space=search_space, model=model, max_evals=2)
        sweep_results = sweep.save()

        # plot performance
        perform_experiments(sweep_results)

    print(basins)