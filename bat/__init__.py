r'''
Black-box Adversarial Toolbox (BAT) is a python library for **Distrubuted Black-box Attacks** against Deep Learning Cloud Services.

## bat.apis

Supported Cloud APIs:

- DeepAPI
- Google Cloud Vision
- Imagga

## bat.attacks

Local Search:

- SimBA Attack [(Guo et al., 2019)](https://arxiv.org/abs/1905.07121)
- Square Attack [(Andriushchenko et al., 2020)](https://arxiv.org/abs/1912.00049)

Gradient Estimation:

- Bandits Attack [(Ilyas et al., 2019)](https://arxiv.org/abs/1807.07978)

## bat.utils

This module implements utility functions.

'''

# Project Imports
from bat import apis
from bat import attacks

# Semantic Version
__version__ = "0.1.0"
