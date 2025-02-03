from dataclasses import dataclass, field
import yaml
from copy import deepcopy

from neuralhydrology.evaluation.metrics import get_available_metrics

from .ParameterSearchSpace import ParameterSearchSpace


@dataclass
class FineTuningArgs:
    
    # Data settings
    
    

    # search space
    search_space: ParameterSearchSpace
    max_evals: int
    
    # Experiment settings
    basin_file: str
    experiment_name: str
    base_model_path: str = None
    output_dir: str = None
    num_workers: int = 1
    validate_every: int = 5
    metrics: list[str] = field(default_factory=get_available_metrics())
    device: int = 'cpu'

    @classmethod
    def from_yaml(cls, file_path: str) -> 'FineTuningArgs':
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert the hyperopt_search_space dictionary to a HyperoptSearchSpace object
        config_dict['search_space'] = ParameterSearchSpace().from_dict(config_dict['search_space'])

        return cls(**config_dict)

    def save(self, path):
        with open(path, 'w') as f:
            loading_self = deepcopy(self)
            loading_self.search_space = self.search_space.to_dict()
            yaml.dump(loading_self.__dict__, f)


