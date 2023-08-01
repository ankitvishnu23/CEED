# CEED
Towards robust and generalizable representations of extracellular data using contrastive learning

## Getting Started
This repo provides tools for training and evaluating a contrastive learning based model for extracellular 
electrophysiology data. 

## Requirements

Your machine has at least one GPU.

## Installation

First create a Conda environment in which this package and its dependencies will be installed.
```console
conda create --name <YOUR_ENVIRONMENT_NAME> python=3.8
```

and activate it:
```console
conda activate <YOUR_ENVIRONMENT_NAME>
```

Move into the folder where you want to place the repository folder, and then download it from GitHub:
```console
cd <SOME_FOLDER>
git clone https://github.com/ankitvishnu23/CEED.git
```

Then move into the newly-created repository folder, and install dependencies:
```console
cd CEED
pip install -r requirements.txt
```

