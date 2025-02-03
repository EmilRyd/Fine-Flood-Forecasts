# do all necessary imports
from dataclasses import dataclass, field
from hyperopt import hp
from typing import Dict, Any

from hyperopt.pyll.stochastic import sample

@dataclass
class ParameterSearchSpace:

    # keep the limits
    data: Dict[str, Any] = None

    # search space
    epochs: Dict[str, Any] = field(default_factory=lambda: hp.quniform('epochs', 1, 40, 1))
    learning_rate: Dict[int, Any] = field(default_factory=lambda: {0: hp.uniform('lr1', 1e-5, 1e-3), 20: hp.uniform('lr2', 1e-6, 1e-4)})
    lstm: Dict[str, Any] = field(default_factory=lambda: hp.choice('lstm', [False, True]))
    loss: Dict[str, Any] = field(default_factory=lambda: hp.choice('loss', ['NSE', 'MSE', 'RMSE']))

    def param_dict_from_model_output(self, best_params: dict):
        args = {}

        for key in self.data.keys():
            match key:
                case 'epochs':
                    args[key] = int(best_params[key])
                case 'learning_rate':
                    args['learning_rate'] = {lr_epoch: float(best_params[f'lr{lr_epoch}']) for lr_epoch in self.learning_rate.keys()}
                case 'loss':
                    args['loss'] = self.data['loss']['options'][best_params['loss']]
                case _:
                    args[key] = best_params[key]

        return args

    def cfg_from_args(self, args) -> dict:
        
        # set dict parameters based on config dictionary passed to function
        data = {}
        for key in args.keys():
            match key:
                case 'epochs':
                    data['epochs'] = int(args['epochs'])
                    data['save_weights_every'] = int(args['epochs'])
                case 'learning_rate':
                    data['learning_rate'] = args['learning_rate']
                case 'loss':
                    data['loss'] = args['loss']
                case 'lstm':
                    modules = ['head'] 
                    if args['lstm']:
                        modules.append('lstm')
                    data['finetune_modules'] = modules
                case _:
                    pass
        
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSearchSpace':
        # Convert the dictionary to Hyperopt search space objects
        
        for key in data.keys():
            match key:
                case 'epochs':
                    epochs = hp.quniform('epochs', data['epochs']['low'], data['epochs']['high'], data['epochs']['q'])
                case 'learning_rate':
                    learning_rate = {
                        step: hp.uniform(f'lr{step}', float(lr_range['low']), float(lr_range['high']))
                        for step, lr_range in data['learning_rate'].items()
                    }
                case 'lstm':
                    lstm = hp.choice('lstm', data['lstm']['options'])
                case 'loss':
                    loss = hp.choice('loss', data['loss']['options'])
                
        '''epochs = hp.quniform('epochs', data['epochs']['low'], data['epochs']['high'], data['epochs']['q'])
        
        learning_rate = {
            step: hp.uniform(f'lr{step}', float(lr_range['low']), float(lr_range['high']))
            for step, lr_range in data['learning_rate'].items()
        }
        
        lstm = hp.choice('lstm', data['lstm']['options'])
        loss = hp.choice('loss', data['loss']['options'])'''
    
        return cls(epochs=epochs, learning_rate=learning_rate, lstm=lstm, loss=loss, data=data)
    
    # implement a to_dict method which reverses to from_dict method
    def to_dict(self) -> Dict[str, Any]:
        return self.data