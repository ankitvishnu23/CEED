# CEED
Towards robust and generalizable representations of extracellular data using contrastive learning (published at NeurIPS 2023)

paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/83c637c3bc0ca88eda6cf4f5f45bdced-Abstract-Conference.html

## Getting Started
This repo provides tools for training, evaluating, and visualizing a contrastive learning based model for extracellular 
electrophysiology data. Tested on **Linux** machines only. 

## Installation

First create a Conda environment in which this package and its dependencies will be installed.
```console
conda create --name <YOUR_ENVIRONMENT_NAME> python=3.10
```

and activate it:
```console
conda activate <YOUR_ENVIRONMENT_NAME>
```

Download CEED from github and then install its dependencies and the package:
```console
git clone https://github.com/ankitvishnu23/CEED.git
cd CEED
pip install -r requirements.txt
pip install -e .
```

## Training and Inference
### Notebooks
Please refer to the respective notebook files in `./notebooks` for generating the data, executing training (on a single GPU), and performing inference and analysis. The notebook files are numbered in order.

### Command-line
Training can also be executed via command-line, for both a single-GPU and multi-GPU set up. 
* For running on a single GPU:
  
```python ./ceed/main.py --data=<path-to-data> --num_extra_chans=5 --arch=fc_encoder --exp=<name-of-expt> ``` 
* For running on a multi-GPU cluster (we use the submitit package on a SLURM cluster)
  
```python ./ceed/launcher.py --data=<path-to-data> --num_extra_chans=5 --arch=scam --exp=<name-of-expt>  ``` 

## CEED model checkpoints and data
To access some example datasets used in the paper and some MLP encoder checkpoints please refer to the following storage link: https://uchicago.box.com/v/CEED-data-storage

