# redirecting the gradients to the inputs

from pathlib import Path

import torch

from experiment import train
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler

from experiment.finetuning.finetune import finetune_model, pick_a_basin, update_files

from experiment.utils import TrainedModelID, get_epoch_string, TrainedModel

def finetune_attributes_for_basin(basin: str, base_model: TrainedModel) -> TrainedModel:
    # create yaml from base_model and basin
    config_file_path = Path(__file__).parent.parent / 'finetuning' / 'assets' / 'attribute_tuning.yml'
    update_files(model=base_model, basin=basin, yml_file_path=config_file_path)

    # finetune the model
    trained_model = finetune_model(config_file_path)
    return trained_model

def fetch_tuned_attributes(trained_model: TrainedModel) -> tuple[torch.Tensor, torch.Tensor]:
    # get the statics corrector tensor
    
    loaded_model = torch.load(str(trained_model.run_dir / f'model_epoch{get_epoch_string(trained_model.cfg.epochs)}.pt'), weights_only=False)
    statics_corrector = loaded_model['embedding_net.statics_corrector'].detach().cpu()

    # get the original static attributes
    scaler = load_scaler(trained_model.run_dir)
    dataset = get_dataset(trained_model.cfg, is_train=False, period='train', scaler=scaler)
    
    assert basin in dataset._attributes.keys(), f'basin ({basin}) not in the dataset attributes ({dataset._attributes})'

    original_attributes = dataset._attributes[basin]

    assert statics_corrector.shape == original_attributes.shape, f'statics corrector and original attributes are not of the same shape, {statics_corrector.shape} and {original_attributes.shape}, respectively'

    final_attributes = statics_corrector + original_attributes

    return final_attributes, original_attributes



    # 
if __name__ == '__main__':
    # pick a base model
    base_model = TrainedModel((TrainedModelID.ATTRIBUTE_TUNING))

    # pick a basin
    basin = pick_a_basin(base_model)

    # train model to finetune attributes
    trained_model = finetune_attributes_for_basin(basin=basin, base_model=base_model)

    # get the new and old attributes to compare
    new_attributes, old_attributes = fetch_tuned_attributes(trained_model=trained_model)




    
    