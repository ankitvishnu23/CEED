#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "black",
    "colorcet",
    "hdbscan",
    "imgaug",
    "matplotlib",
    "opencv-python",
    "pandas",
    "pillow",
    "scikit-image",
    "scikit-learn",
    "spikeinterface",
    "tensorboard_logger",
    "tensorboard",
    "torchtyping",
    "torchvision",
    "typeguard",
    "umap-learn",
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
