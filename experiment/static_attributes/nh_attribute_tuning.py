# redirecting the gradients to the inputs

from pathlib import Path

from neuralhydrology.nh_run import finetune

if __name__ == '__main__':
    config_file_path = Path(__file__).parent.parent / 'finetuning' / 'assets' / 'attribute_tuning.yml'
    model = finetune(config_file_path)
