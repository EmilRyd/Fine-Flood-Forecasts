what i want people to be able to use this repository for:

input: config file with fine-tuning parameters
output: finetuned_model
requires: all the caravan data in a data/ folder (specified in README.md), some libraries installed (specific in requirements.txt)
does: loads pre-trained model (from hugging face or elsewhere), performs fine-tuning using the config, then saves the finetuned model

structure:
config.yml file contains all the parameters for the sweep (max_evals, lrs, basin_file, base_model, etc). only 1 config file.

code takes this config and does the sweep.

5. Fine-tune on your data: Follow the instructions in our repository at repo to
fine-tune your pre-trained model on your chosen basins. That’s it!

