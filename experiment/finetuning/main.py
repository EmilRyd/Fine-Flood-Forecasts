# function to run automated finetuning on a basin
from experiment.finetuning.finetune import pick_a_basin, find_best_finetuning_params
from hyperopt import hp
from experiment.utils import TrainedModel, TrainedModelID
from experiment.finetuning.experiments import perform_experiments

 # select base model, start with SOTA Camels model
model = TrainedModel(TrainedModelID.SOTA_20)

# define hyperparameter search space
search_space = {
    # TODO: sort out this 5 in finetune(args) code
    'epochs': 5 + hp.randint('epochs', 5),
    'learning_rate': {0: hp.uniform('lr1', 1e-4, 1e-3), 5: hp.uniform('lr2', 1e-5, 1e-4)},
    'lstm': hp.choice('lstm', [False, True]),
    'loss': hp.choice('loss', ['NSE', 'RMSE', 'MSE'])
}

def finetune_on_n_basins(n=500):
    basins = []
    sweeps = []
    for _ in range(n):
        # pick basin
        basin, _ = pick_a_basin(model=model)
        basins.append(basin)
        search_space['basin'] = basin

        # finetune a model  
        sweep = find_best_finetuning_params(search_space=search_space, model=model, max_evals=25)
        sweep_results = sweep.save()
        sweeps.append(sweep_results)
    

    return basins, sweeps
if __name__ == '__main__':

    # actual experiment (I think)

    # pick 3 basins

    # for each basin, do the following
        # finetune a model using some set search_space (how do you make sure you finetune a model really well? that should likely be done through a notebook, or should it be done manually?)

        # print out the test set comparison for that basin
        # print out the test set comparison for across all basins
        # plot the validation and training loss for the base model (already stored in output.log), and the validation and training loss for the finetuning model (already stored in output.log)
    basins, sweeps = finetune_on_n_basins()
    for idx, sweep in enumerate(sweeps):
        # plot performance
        perform_experiments(sweep)
        print(basins[idx])
    