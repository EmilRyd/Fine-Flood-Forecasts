# Fine-Flood-Forecasts: Incorporating local data into global models through fine-tuning

This repository is meant for anyone who wants to fine-tune a global ML-based streamflow prediction model on local data. More details in our paper (submitted to the AI for Climate Change workshop at ICLR 2025).

## How to use this repository: a step-by-step guide on fine-tuning

### 1. Download Caravan

First, you must download the [Caravan](https://github.com/kratzert/Caravan) dataset from [here](https://zenodo.org/records/14673536). There are more details in the Caravan [Github repository](https://github.com/kratzert/Caravan). The dataset is ~100GB in size.

### 2. Clone the repository
Clone the repository, by navigating to the folder on your local computer that you want to work in, and type (in the terminal):
```
git init 
git clone url/to/this/repo
```

Then, add move the Caravan data into a folder in your local project. Whatever folder you choose to put it in, make sure to add this into the field ```data_dir: ``` in the config file when you run the fine-tuning script in step 4.

Activate the right python environment by running:
```
conda env create -f environment.yml
```

To run this command you will need Anaconda or miniconda, which can be installed [here](https://www.anaconda.com/download). If you prefer using virtualenv or and environment manager, you can just copy the requirements from the environment.yml file.

To activate your environment (necessary for next step), run:

```
conda activate finetuning
```

### 3. Fine-tune a pre-trained model on your basins
To fine-tune a pre-trained model on the basins of your choosing. Run:
```
python scripts/finetune.py config=path/to/config/file
```

See the ```configs``` folder for an example config file. It is in your config file that you can make all the choices regarding your fine-tuning run. Things to set in your config file:

1. ```base_model_path```: This is the path to your pre-trained model. If you do not have one, don't worry! If you leave this field empty, the code will automatically download a pre-trained model from Hugging Face ([link to model repo]([url](https://huggingface.co/EmilRyd/caravan_model))) onto your computer that you can use. After it is downloaded, you should set base_model_path to the path in which the model is downloaded ```pretrained_models/caravan_base```, so you don't need to re-download the model each time.
2. Hyperparameters for your finetuning sweep in your config file.
3. What basins to fine-tune on (by picking the ```basin_file```). You can edit the standard basin file in ```assets/basins.txt``` to include exactly the basins you want to add. You can find all the basins in [Caravan](https://github.com/kratzert/Caravan) (and since you have the Caravan dataset on your computer you can readily print out all the basins).

The outputs from the fine-tuning sweep will end up in the ```output/``` folder. This folder will contain 3 subfolders:
1. ```finetuned_models``` containing the best fine-tuned models for each basin.
2. ```runs```containing all the attempted fine-tuned models for each basin (same as the number of ```max_evals```in your config, per basin.
3. ```sweeps``` containing a .pkl file for every basin you fine_tune on, with information regarding the hyperparameter sweep.

### 4. Analyze your results
When you are done fine-tuning, you can analyze your results by running the ```analysis.ipynb```notebook in the ```analysis``` folder. This notebook contains only the mest rudimentary display of the results, and there are far more things you may want to explore, such as the relationship between fine-tuning improvements and hydrological attributes of the basins.

### 5. (Optional) Format your own data into Caravan to fine-tune on
In the steps described above, you are limited to fine-tuning on the 22,732 basins that are already in the Caravan dataset. If you have your own basins (with accompanying catchment delineations and streamflow data), you can format them to fit into the Caravan dataset by following the tutorials here. You can then either contribute them to the Caravan dataset itself (it's very easy), or you can just keep them as a local copy without having to publicize your data. Either way, you can then add your basin ids into the basins.txt file just as any other basin in Caravan, and then run the same fine-tuning procedure as above. Voilà!

## Questions/problems
If you have any comments or problems with the repository, feel free to open an issue! You can also reach me at emil.ryd@new.ox.ac.uk.
