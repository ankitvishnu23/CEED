# CEED
Towards robust and generalizable representations of extracellular data using contrastive learning

## Getting Started
This repo provides tools for training, evaluating, and visualizing a contrastive learning based model for extracellular 
electrophysiology data. 

## Installation

First create a Conda environment in which this package and its dependencies will be installed.
```console
conda create --name <YOUR_ENVIRONMENT_NAME> python=3.10
```

and activate it:
```console
conda activate <YOUR_ENVIRONMENT_NAME>
```

Download CEED from github and then install dependencies:
```console
git clone https://github.com/ankitvishnu23/CEED.git
cd CEED
pip install -r requirements.txt
```
Once this is done, you will need to download and install the dartsort package, which is used for data generation procedures:
```console
git clone https://github.com/cwindolf/dartsort.git
cd dartsort
pip install -r requirements.txt
```

### For macOS users only:
To use the ibllib package, which is involved in dataset generation, Qt5 must be installed separate from pip. 
We will use homebrew to install it (install at https://brew.sh/ if you have not already) using the following command:
```console
brew install qt5
brew link qt5 --force
```

followed by a command that will look similar to the following, and will show up on the terminal upon installation:
```console
echo 'export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"' >> ~/.zshrc
```

finally, run the following command, which may take a while to complete:
```console
pip install pyqt5 --config-settings --confirm-license= --verbose
```

Download CEED from github and then install dependencies:
```console
git clone https://github.com/ankitvishnu23/CEED.git
cd CEED
pip install -r requirements.txt
```

Once this is done, you will need to download and install the dartsort package, which is used for data generation procedures:
```console
git clone https://github.com/cwindolf/dartsort.git
cd dartsort
pip install -r requirements.txt
```
