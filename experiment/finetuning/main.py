# function to run automated finetuning on a basin
from hyperopt import hp

from experiment.utils import TrainedModel, TrainedModelID
from experiment.finetuning.finetune import finetune_on_n_basins
from experiment.finetuning.experiments import run_all_experiments


if __name__ == '__main__':

    # actual experiment (I think)
    # select base model, start with SOTA Camels model
    model = TrainedModel(TrainedModelID.SOTA_20)

    # define hyperparameter search space
    search_space = {
        'epoch_offset': 1,
        'additional_epochs': hp.randint('epochs', 2),
        'learning_rate': {0: hp.uniform('lr1', 1e-5, 5e-5), 5: hp.uniform('lr2', 1e-6, 1e-5)},
        'lstm': hp.choice('lstm', [False, True]),
        'loss': hp.choice('loss', ['NSE', 'RMSE', 'MSE'])
    }

    basins, sweeps, run_dir = finetune_on_n_basins(model=model, search_space=search_space, max_evals=3)
    run_all_experiments(basins=basins, run_dir=run_dir)

    # pick 3 basins

    # for each basin, do the following
        # finetune a model using some set search_space (how do you make sure you finetune a model really well? that should likely be done through a notebook, or should it be done manually?)

        # print out the test set comparison for that basin
        # print out the test set comparison for across all basins
        # plot the validation and training loss for the base model (already stored in output.log), and the validation and training loss for the finetuning model (already stored in output.log)
    