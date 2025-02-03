import os
from pathlib import Path
# finetuning experiment code


from utils import TrainingExperimentResults, load_pkl, filter_outliers, LOSSES
import matplotlib.pyplot as plt

from utils import TrainedModel, read_txt_to_list
import pandas as pd
import pickle as p

from tqdm import tqdm

class FinetuningStatistics:

    
    def __init__(self, sweep_name: str, experiment_path: str, basins: list[str]):
        
        self.basins = basins
        self.losses = LOSSES
        
        self.sweep_name: str = sweep_name

        self.sweep_path =  experiment_path / 'sweeps' 

        

    def get_statistics(self, metric='NSE', load_from_cache=True):
        # get the sweeps
        print('Loading sweeps')
        self.get_sweeps(load_from_cache)

        # get the evaluation statistics
        print('Evaluating performance')
        df = self.evaluate_performance(metric=metric, load_from_cache=load_from_cache)

        return df
            
    def evaluate_performance(self, metric: str, load_from_cache):
        #full_delta_dict = {'val_basin': [], 'test_basin': [], 'test_all': [], 'test_origin': [], 'val_origin': [], 'base_model': []}
        
        df = pd.DataFrame()


        # get the base models and base vals
        # when you have the sweeps
        base_paths = list(set([sweep.base_model.run_dir / 'config.yml' for sweep in self.sweeps]))
        base_models = [TrainedModel(base_path) for base_path in base_paths]  
        
        
        # assuming its all the same base function (safe for now)
        #base_vals = {str(base_model.run_dir): evaluate_model_full(base_model, basins=self.basins, period='validation', ignore_previous_metrics=False) for base_model in base_models}
        #base_tests = {str(base_model.run_dir): evaluate_model_full(base_model, basins=self.basins, period='test', ignore_previous_metrics=False) for base_model in base_models}
        base_vals = {str(base_model.run_dir):  base_model.get_eval_metrics(period='validation', basins=self.basins) for base_model in base_models}
        base_tests = {str(base_model.run_dir): base_model.get_eval_metrics(period='test', basins=self.basins) for base_model in base_models}

        # iterate over each model individually
        for base_model in tqdm(base_models):
            model_sweeps = [sweep for sweep in self.sweeps if sweep.base_model.run_dir==base_model.run_dir]
            model_df = self.performance_comparison_for_model(model_sweeps = model_sweeps, base_val=base_vals[str(base_model.run_dir)], base_test=base_tests[str(base_model.run_dir)], 
                                                        base_model=base_model, metric=metric, using_only_evaluated_models=True)
            df = pd.concat([df, model_df], ignore_index=True)

        
        
        self.plot_deltas(df.select_dtypes(include='float'), metric=metric)
        return df

    # get sweeps from all run dirs
    def get_sweeps(self, load_from_cache) -> list[TrainingExperimentResults]:
        
        # reset sweeps list
        self.sweeps = []

        
        self.sweeps = [load_pkl(sweep_file) for sweep_file in self.sweep_path.iterdir() if sweep_file.stem in self.basins]
        
        

    def performance_comparison_for_basin(self, sweep: TrainingExperimentResults, base_val: pd.DataFrame, base_test: pd.DataFrame, metric: str):
        # load the sweep
        df = pd.DataFrame(columns='val_delta test_delta test_origin val_origin basin'.split())
        fine_validation_score = -min(sweep.trials.losses())
        final_model = getattr(sweep, 'final_model', None) or getattr(sweep, 'finetuned_model', None)

        # sanity check on the validation losses
        
        fine_val_basin = final_model.get_eval_metrics(period='validation', basins=[sweep.basin])
        
        assert len(fine_val_basin[metric].values) == 1, f'expected only score for 1 basin, got {len(fine_val_basin[metric].values)} scores'
        assert fine_val_basin[LOSSES[sweep.best_params['loss']]].values[0] == fine_validation_score, "final validation score not as expected"
        
        fine_metric_score = float(fine_val_basin[metric].values[0])

        base_test_basin = base_test[base_test.basin==sweep.basin]

        #fine_test_basin = evaluate_model_full(model=final_model, basins=[sweep.basin], period='test', ignore_previous_metrics=False)    
        fine_test_basin = final_model.get_eval_metrics(period='test', basins=[sweep.basin])    
        
        # for individual basin

        # validation
        base_val_score = float(base_val[base_val.basin==sweep.basin][metric].iloc[0])
        val_delta_basin = fine_metric_score - base_val_score
        
        # test
        fine_test_score = float(fine_test_basin.iloc[0][metric])
        base_test_score = float(base_test_basin.iloc[0][metric])
        test_delta_basin = fine_test_score - base_test_score

        df.loc[0] = [val_delta_basin, test_delta_basin, base_test_score, base_val_score, sweep.basin]
        
        return df

    def performance_comparison_for_model(self, model_sweeps: list, base_val: pd.DataFrame, base_test: pd.DataFrame, base_model: TrainedModel, metric: str, using_only_evaluated_models=True):
        # iterate over the sweeps and plot the comparison
        model_df = pd.DataFrame()
        
        for sweep in model_sweeps:
            # return if no eval available
            final_model = getattr(sweep, 'final_model', None) or getattr(sweep, 'finetuned_model', None)

            metrics_file = final_model.get_eval_metrics_file(period='test')
            if os.path.exists(metrics_file) or not using_only_evaluated_models:
        
                basin_df = self.performance_comparison_for_basin(sweep, base_val=base_val, base_test=base_test, metric=metric)
                model_df = pd.concat([model_df, basin_df], ignore_index=True)
        
        # add the base model
        model_df['base_model'] = os.path.basename(base_model.run_dir)
        return model_df
      
    
    def plot_deltas(self, df, metric):
        means = df.mean()
        stds = df.std()
        medians = df.median()
        for col in df.columns: 
            if col.endswith('_delta'):
                ave = means[col]
                std = stds[col]
                median = medians[col]

                # filter out the outliers
                no_outliers_data = filter_outliers(df[col].values)
                plt.hist(no_outliers_data, bins=50)
                
                nicename: str = col.split('_')[0] + ' set change'
                plt.title(f'{nicename} in {metric}: average: {ave:.3f} +- {std:.3f}, median: {median:.3f}')
                plt.show()