
import yaml
from pathlib import Path
import os
import shutil
import tempfile

from hyperopt import fmin, Trials, tpe, STATUS_OK
from hyperopt.exceptions import AllTrialsFailed
import pandas as pd

from neuralhydrology.nh_run import finetune, eval_run
from neuralhydrology.training import LOGGER
from neuralhydrology.utils.errors import NoTrainDataError

from utils import EPOCH_SWITCH, LOSSES, TrainedModel, get_epoch_string, read_txt_to_list, TEMP_DIR, TrainingExperimentResults, OUTPUT_DIR, load_hf_model
from .FineTuningArgs import FineTuningArgs

class FineTuningExperiment:


    def __init__(self, experiment_args: FineTuningArgs):
        
        self.search_space = experiment_args.search_space

        self.experiment_name = experiment_args.experiment_name
        if experiment_args.output_dir:
            self.output_dir = Path(experiment_args.output_dir)
        else:
            # standard output dir
            self.output_dir = OUTPUT_DIR
            os.makedirs(self.output_dir, exist_ok=True)
        self.experiment_dir = self.output_dir / self.experiment_name
        self.best_models_dir = self.experiment_dir / 'finetuned_models'

        self.basin_file = Path(experiment_args.basin_file)
        self.basins = read_txt_to_list(self.basin_file)

        assert self.basins, 'No basins found in basin file'

        self.basin = self.basins[0]
        print(self.basins)
        self.max_evals = experiment_args.max_evals
        
       
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir, exist_ok=True)
            
            # write config to output_dir
            experiment_args.save(self.experiment_dir / 'config.yml')
            
            
        self.run_dir = self.experiment_dir / 'runs'
        self.sweeps = []
        self.train_file = None # this will get changed later
        
        # Add the path to the pre-trained model to the finetune config
        self.basin_run_dir = self.run_dir
        self.experiment_counter = 0
        
        # set up base model
        if experiment_args.base_model_path:
            self.base_model = TrainedModel(Path(experiment_args.base_model_path) / 'config.yml')
        else:
            self.base_model = load_hf_model()
                    
        # this will be filled in the set config class
        # TODO fix handling None inputs
        self.finetune_config = {}
        self.finetune_config['base_run_dir'] = str(self.base_model.run_dir.absolute()) # set base run dir, where the pre-trained model is stored
        self.finetune_config['num_workers'] = experiment_args.num_workers
        self.finetune_config['metrics'] = experiment_args.metrics
        self.finetune_config['validate_every'] = None

        self.sweep_dir = self.experiment_dir / 'sweeps'

        
    def run(self, retrain=False) -> tuple[list[Path], Path]:
        ''' run a finetuning search on all untouched (not already finetuned) basins in the experiment '''

        
        if os.path.exists(self.sweep_dir):
            LOGGER.info(f'Experiment {self.experiment_name} already exists, values may be overridden')
        os.makedirs(self.sweep_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        # check basins that already have a trained model
        trained_basins = [f.name for f in self.run_dir.iterdir() if f.is_dir() and len(os.listdir(f)) == self.max_evals + 1]
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        for basin in self.basins:
            if not basin in trained_basins or retrain:
                self.experiment_counter = 0
                self.basin = basin
                self.update_files()

                # finetune a model
                try:  
                    sweep = self.find_best_params()
                    
                    sweep_results = sweep.save(run_dir=self.sweep_dir)
                    self.sweeps.append(sweep_results)
                except NoTrainDataError:
                    LOGGER.warning(f'No training data for basin {basin}')
                except AllTrialsFailed:
                    LOGGER.warning(f'All trials failed for basin {basin}')

        if set(self.basins).issubset(set(trained_basins)) and not retrain:
            print('All basins already trained, set retrain=True to retrain')
        # clean up
        shutil.rmtree(TEMP_DIR)

        print(f'All {len(self.basins)} basins finetuned, finetuned models stored in {self.best_models_dir}')
        return self.sweeps, self.sweep_dir

    def update_files(self):

        basin_dir = self.basin_run_dir / f'{self.basin}' # think this is the right path, not verified
        
        os.makedirs(basin_dir, exist_ok=True) # if not even any general run dir


        # TODO implement this so that instead of a base finetune file, data is just loaded here from the args
        '''# Load the existing YAML data
        with open(self.train_file, 'r') as f:
            data = yaml.safe_load(f)'''
        
        # create a temporary basin file in TEMP dir
        basin_file_path = TEMP_DIR / f'basin.txt'
        with open(basin_file_path, 'w') as fp:
            fp.write(self.basin)
        
        #basin_file_path = get_basin_file(basin)
        self.finetune_config['train_basin_file'] = str(basin_file_path.absolute())
        self.finetune_config['validation_basin_file'] = str(basin_file_path.absolute())
        self.finetune_config['test_basin_file'] = str(basin_file_path.absolute())
        self.finetune_config['run_dir'] = str(basin_dir.absolute()) 

        # Create a basin file with the basin we selected above
        with open(basin_file_path, 'w') as fp:
            fp.write(self.basin)
        
    def param_dict_from_model_output(self, best_params: dict) -> dict:
        # TODO this should be in the dataclass I think

        args = {}
        args['epochs'] = int(best_params['epochs'])
        args['learning_rate'] = {0: float(best_params['lr1']), EPOCH_SWITCH: float(best_params['lr2'])}
        args['loss'] = LOSSES[best_params['loss']]
        args['lstm'] = best_params['lstm']
        return args 
    

    def cfg_from_args(self, args) -> dict:
        
        # set dict parameters based on config dictionary passed to function
        
        data = {}
        modules = ['head'] 
        if args['lstm']:
            modules.append('lstm')
        data['epochs'] = int(args['epochs'])
        data['learning_rate'] = args['learning_rate']
        data['loss'] = args['loss']
        data['finetune_modules'] = modules
        data['save_weights_every'] = int(args['epochs'])
   
        return data

    def train_model_from_cfg(self, data: dict):
        # finetune using temporary yaml file
        
        with tempfile.NamedTemporaryFile(delete=True, dir=TEMP_DIR, suffix='.yml', mode='w') as f:
            # add the base finetuning config args to the chosen hyperparams
            data.update(self.finetune_config)

            # load into file
            yaml.dump(data, f)  

            # finetune using neuralhydrology function
            finetune(TEMP_DIR / f.name)
            

        # TODO check this file
        basin_dir = Path(data['run_dir']) / data['experiment_name']
        config_file_path = basin_dir / 'config.yml'

        trained_model = TrainedModel(config_file_path_or_experiment_name=config_file_path)

        # find eval score
        eval_run(basin_dir, period='validation')
        v_df = pd.read_csv(basin_dir / 'validation' / f'model_epoch{get_epoch_string(data["epochs"])}' / 'validation_metrics.csv')
        

        # assert that metric is available
        assert data['loss'] in v_df.columns, 'loss not in validation metrics'
        # return negative validation score
        return {'loss': -float(v_df[data['loss']].values[0]), 'status': STATUS_OK, 'model': trained_model}

    def train_model(self, args, best_model=False):
        # will be redefined in child class
        data = self.cfg_from_args(args)
        if best_model:
            data['experiment_name'] = f'{self.basin}'  
        else:
            data['experiment_name'] = f'{self.experiment_counter}'  
        self.experiment_counter += 1
        score = self.train_model_from_cfg(data=data)
        return score
    
    def find_best_params(self) -> TrainingExperimentResults:
    
        trials = Trials()
    
        best_params = fmin(self.train_model, space=self.search_space.__dict__, algo=tpe.suggest, max_evals=self.max_evals, trials=trials, verbose=True)
        
        # add basin back to best params    
        # run best model to get that fresh validation data
        best_args = self.search_space.param_dict_from_model_output(best_params)

        # update the files to be for best model
        self.finetune_config['run_dir'] = str(self.best_models_dir.absolute())
        
        training_data = self.train_model(best_args, best_model=True)
        trained_model = training_data['model']

        # store model, finetuned model, and best_params
        sweep = TrainingExperimentResults(best_params=best_params, base_model=self.base_model,
                    final_model=trained_model, search_space=self.search_space,
                        max_evals=self.max_evals, basin=self.basin, trials=trials)

        # perform evaluation on the test set
        eval_run(run_dir=sweep.final_model.run_dir, period='test')
        
        return sweep
