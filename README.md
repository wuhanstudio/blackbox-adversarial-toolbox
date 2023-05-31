<img src="https://bat.wuhanstudio.uk/images/bat.png" width=300px style="float: left;" >

# Black-box Adversarial Toolbox (BAT)


[![Build Status](https://app.travis-ci.com/wuhanstudio/blackbox-adversarial-toolbox.svg?branch=master)](https://app.travis-ci.com/wuhanstudio/blackbox-adversarial-toolbox)
[![PyPI version](https://badge.fury.io/py/blackbox-adversarial-toolbox.svg)](https://badge.fury.io/py/blackbox-adversarial-toolbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/blackbox-adversarial-toolbox)](https://pypi.org/project/blackbox-adversarial-toolbox/)

A Python Library for Deep Learning Security that focuses on Distributed Black-box attacks.


## Installation

```python
pip install blackbox-adversarial-toolbox
```


## Usage (CLI)

```
Usage: bat [OPTIONS] COMMAND [ARGS]...

  The CLI tool for Black-box Adversarial Toolbox (BAT).

Options:
  --help  Show this message and exit.

Commands:
  api      Manage Cloud APIs
  attack   Manage Attacks
  example  Manage Examples
```
Useful commands:
```
# List supported Cloud APIs
bat api list

# List suported Attacks
bat attack list

# Test Cloud APIs
bat api run deepapi
bat api run google
bat api run imagga

# Run exmaples
bat example run simba_deepapi
bat example run bandits_deepapi
bat example run square_deepapi
```

## Usage (Python)

```python
import numpy as np
from PIL import Image

from bat.attacks import SimBA
from bat.apis.deepapi import DeepAPI_VGG16_Cifar10

# Load Image
x = np.asarray(Image.open("dog.jpg").convert('RGB'))
x = np.array([x])

# Initialize the Cloud API Model
DEEP_API_URL = 'http://localhost:8080'
model = DeepAPI_VGG16_Cifar10(DEEP_API_URL)

# Get Preditction
y_pred = model.predict(x)[0]

# Distributed SimBA Attack
simba = SimBA(model)
x_adv = simba.attack(x, np.argmax(y_pred), epsilon=0.05, max_it=10)
```

<h3> <a href="https://bat.wuhanstudio.uk/"> Documentation </a>
