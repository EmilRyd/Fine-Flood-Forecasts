# redirecting the gradients to the inputs

from pathlib import Path

import torch

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler

from experiment.utils import TrainedModelID, get_epoch_string, TrainedModel, load_camels_basins

from experiment.finetuning.finetune import finetune_model, update_files


def finetune_attributes_for_basin(basin: str, base_model: TrainedModel) -> TrainedModel:
    # create yaml from base_model and basin
    config_file_path = Path(__file__).parent / 'assets' / 'attribute_tuning.yml'
    update_files(model=base_model, basin=basin, yml_file_path=config_file_path)

    # finetune the model
    trained_model = finetune_model(config_file_path)
    return trained_model

def fetch_tuned_attributes(trained_model: TrainedModel, basin: str) -> tuple[torch.Tensor, torch.Tensor]:
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

def finetune_attributes(basins: list, base_model: TrainedModel)-> dict[str|TrainedModel]:
    tuned_models = {}
    
    for basin in basins:
        tuned_model = finetune_attributes_for_basin(basin, base_model=base_model)
        tuned_models[basin] = tuned_model
        
    
    return tuned_models

def examine_attributes(tuned_models: list[TrainedModel]):
    attribute_deltas = []
    for basin, tuned_model in tuned_models.items():
        new_attributes, old_attributes = fetch_tuned_attributes(trained_model=tuned_model, basin=basin)
        delta = new_attributes - old_attributes
        attribute_deltas.append(delta)
    


    # examine the attribute deltas in some way
    print(attribute_deltas)




if __name__ == '__main__':
    # pick a base model
    base_model = TrainedModel((TrainedModelID.ATTRIBUTE_TUNING))

    # pick a basin
    basins = load_camels_basins()[:2]

    # train models to finetune attributes
    tuned_models= finetune_attributes(basins=basins, base_model=base_model)

    # get the new and old attributes to compare
    examine_attributes(tuned_models)
    




    
    