# finetuning experiment code
from pathlib import Path
import pickle as p
from IPython.core.display import display, HTML

from experiment.eval import evaluate_models

from experiment.finetuning.finetune import cfg_from_args
from experiment.finetuning.utils import generate_sweep_run_directory, load_pkl, Sweep, get_training_losses, param_dict_from_model_output
import matplotlib.pyplot as plt

import os
from experiment.utils import TrainedModel, TrainedModelID
import pandas as pd
import numpy as np

def show_performance_comparison(models: list, basins: list = []):
    if len(basins) == 0:
        # compare how the model performs on all basins
        df = evaluate_models(models=models, bolden_values=True, include_benchmark=False)
    else:
        # compare finetuned model and model's performance on the selected basin
        df = evaluate_models(models=models, basins=basins, bolden_values=True, include_benchmark=False)

    # display the comparison
    display(HTML(data=df.to_html(escape=False)))

def plot_metrics(sweeps: list):

    for sweep in sweeps:
        # experiment #1, compare across all basins
        show_performance_comparison(models=[sweep.base_model, sweep.finetuned_model])

        #maybe comapre the validatio loss improvement as welL, for sanity


        # experiment #0, compare for the finetuned basin
        show_performance_comparison(models=[sweep.base_model, sweep.finetuned_model], basins=[sweep.basin])
        
    # experiment #2, compare the ratios of training and validation over the training and the finetuning period    
def performance_comparison_for_basin(sweep: Sweep):
    # load the sweep
    fine_validation_score = -min(sweep.trials.losses())
    
    base_val = evaluate_models([sweep.base_model], basins=[sweep.basin], include_benchmark=False, period='validation', ignore_previous_metrics=False)
    
    # sanity check on the validation losses
    fine_val_basin = evaluate_models([sweep.finetuned_model], basins=[sweep.basin], include_benchmark=False, period='validation', ignore_previous_metrics=False)
    assert float(fine_val_basin.iloc[0][f'basin_{sweep.basin}']) == fine_validation_score, f'finetuned model final validation score ({float(fine_val_basin.iloc[0][f"basin_{sweep.basin}"])}) and negative best loss ({fine_validation_score}) do not equal each other'
    
    base_test = evaluate_models([sweep.base_model], include_benchmark=False, period='test', ignore_previous_metrics=False)
    base_test_basin = evaluate_models([sweep.base_model], basins=[sweep.basin], include_benchmark=False, period='test', ignore_previous_metrics=False)
    
    fine_test = evaluate_models([sweep.finetuned_model], include_benchmark=False, period='test', ignore_previous_metrics=False)
    fine_test_basin = evaluate_models([sweep.finetuned_model], basins=[sweep.basin], include_benchmark=False, period='test', ignore_previous_metrics=False)

    
    # for all basins (only test set)
    fine_test_score = float(fine_test.iloc[0][f'basin_{sweep.basin}'])
    base_test_score = float(base_test.iloc[0][f'{sweep.base_model.config_id}'])
    test_delta_all = fine_test_score - base_test_score
    if test_delta_all > 0:
        print(test_delta_all)
    # for individual basin

    # validation
    val_delta_basin = fine_validation_score - float(base_val.iloc[0][f'{sweep.base_model.config_id}'])
    
    # test
    fine_test_score = float(fine_test_basin.iloc[0][f'basin_{sweep.basin}'])
    base_test_score = float(base_test_basin.iloc[0][f'{sweep.base_model.config_id}'])
    test_delta_basin = fine_test_score - base_test_score

    return {'val_basin': val_delta_basin, 'test_basin': test_delta_basin, 'test_all': test_delta_all}

def performance_comparison(sweeps: list):
    # iterate over the sweeps and plot the comparison
    delta_dict = {'val_basin': [], 'test_basin': [], 'test_all': []}
    for sweep in sweeps:
        deltas = performance_comparison_for_basin(sweep)
        for key, value in delta_dict.items():
            if not deltas[key] > 1000:
                delta_dict[key] = value + [deltas[key]]
    for key, value in delta_dict.items():
        
        plt.hist(value)
        ave = np.mean(value)
        median = np.median(value)
        std = np.std(value)
        plt.title(f'{key}, average delta: {ave:.2f} +- {std:.2f}, median delta: {median:.2f}')
        plt.show()

def get_loss_ratios(model: TrainedModel) -> pd.Series:

     # maybe this should go into a more general utils file?
    train_df, val_df = get_training_losses(model)
    if not val_df.empty:
        val_df.rename(columns={'value': 'val'}, inplace=True)
        train_df.rename(columns={'value': 'train'}, inplace=True)

        ratio_df = pd.merge(left=train_df, right=val_df, on='step')
        ratios = ratio_df.val/ratio_df.train
        ratios = ratios.set_axis(ratio_df['step'])
        return ratios
    else:
        return pd.DataFrame()

def loss_ratio_comparison(sweeps: list, use_base=False):
    # compare the loss ratio of the base model with the loss ratios from the finetuning
    
    if use_base:
        # assuming the same base model
        assert len(set([sweep.base_model.config_id for sweep in sweeps])) == 1, f'more than 1 unique base model ({len(set([sweep.base_model.config_id for sweep in sweeps]))} to be exact) for finetuned models. This behavior is not accounted for yet in implementation'

        # find the ratio for the base model
        base_model = sweeps[0].base_model
    else:
        base_model = TrainedModel(TrainedModelID.SOTA)    
    base_ratios = get_loss_ratios(model=base_model)
    plt.plot(base_ratios.index, base_ratios, label='base', c='b')
    fine_ratios = pd.DataFrame()
    for sweep in sweeps:
        fine_ratio_series = get_loss_ratios(sweep.finetuned_model)
        
        # shift by the base_model number of epochs
        if not fine_ratio_series.empty:
            fine_ratio_series.loc[0] = base_ratios.loc[len(base_ratios)-1]

            fine_ratio_series.index = fine_ratio_series.index + base_ratios.index.max()
            fine_ratio_series = fine_ratio_series.sort_index()
            fine_ratios = pd.concat([fine_ratios, fine_ratio_series], axis=1)
            plt.plot(fine_ratio_series.index, fine_ratio_series, alpha=0.2, c='k')

    ave_ratios = fine_ratios.mean(axis=1).sort_index()
    plt.plot(ave_ratios.index, ave_ratios, label='average of finetuned models', c='r')
    plt.legend()

    plt.show()

def plot_fine_parameters(sweeps: list):
    lrs_0 = []
    lrs_5 = []
    modules = []
    epochs = []
    for sweep in sweeps:
        params = sweep.best_params
        lrs_0.append(params['lr1'])
        epochs.append(params['epochs'])
        lrs_5.append(params['lr2'])
        modules.append(params['lstm'])
    plt.hist(lrs_0)
    plt.title('lr1')
    plt.show()
    plt.hist(lrs_5)
    plt.title('lr2')
    plt.show()
    plt.hist(modules)
    plt.title('lstm or not')
    plt.show()
    plt.hist(epochs)
    print(epochs)
    plt.title('epochs')
    plt.show()

def get_sweeps(basins: list, run_dir: Path) -> list[Sweep]:
    sweeps = []
    
    if len(basins) == 0:
        
        for filename in os.listdir(run_dir):
            sweep_results = Path(os.path.join(run_dir, filename))
            if os.path.isfile(sweep_results):
                sweep = load_pkl(sweep_results)
                sweeps.append(sweep)
    else:
        for basin in basins:
            sweep_results = run_dir / f'{basin}.pkl'
            sweep = load_pkl(sweep_results)
            sweeps.append(sweep)
    return sweeps

# reading resluts and performing experiments on them
def run_all_experiments(run_dir: Path, basins: list = []):
    sweeps = get_sweeps(basins=basins, run_dir=run_dir)
    performance_comparison(sweeps)
    loss_ratio_comparison(sweeps)

    plot_fine_parameters(sweeps=sweeps)
    if basins:
        plot_metrics(sweeps)



if __name__ == '__main__':
    # sanity check on the finetuned validation losses
    
    # get sweeps
    run_dir = Path(__file__).parent / 'sweeps' / 'sota_20'
    run_all_experiments(run_dir=run_dir)