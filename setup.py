#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "black",
    "brainbox",
    "colorcet",
    "classy_vision==0.7.0",
    "cloudpickle",
    "dartsort @ git+https://github.com/cwindolf/dartsort.git",
    "hdbscan",
    "ibllib",
    "imgaug",
    "ipykernel",
    "matplotlib",
    "MEArec",
    "ONE-api",
    "opencv-python",
    "pandas",
    "pillow",
    "scikit-image",
    "scikit-learn",
    "spikeinterface",
    "tensorboard_logger",
    "tensorboard",
    "tensorflow",
    "torchtyping",
    "typeguard",
    "umap-learn",
    "pyfftw",
]


setup(
    name="CEED",
    packages=find_packages(),
    version=version,
    description="contrastive learning for learning spike features",
    author="Ankit Vishnu",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="av3016@columbia.edu",
)
