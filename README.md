<img src="https://bat.wuhanstudio.uk/images/bat.png" width=300px style="float: left;" >

<h3> <a href="https://bat.wuhanstudio.uk/"> Documentation </a>

# Black-box Adversarial Toolbox (BAT)


[![Build Status](https://app.travis-ci.com/wuhanstudio/blackbox-adversarial-toolbox.svg?branch=master)](https://app.travis-ci.com/wuhanstudio/blackbox-adversarial-toolbox)
[![PyPI version](https://badge.fury.io/py/blackbox-adversarial-toolbox.svg)](https://badge.fury.io/py/blackbox-adversarial-toolbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/blackbox-adversarial-toolbox)](https://pypi.org/project/blackbox-adversarial-toolbox/)

A Python Library for Deep Learning Security that focuses on Black-box attacks.

## Installation

```python
pip install blackbox-adversarial-toolbox
```



## Usage



```python
import numpy as np
from PIL import Image

from bat.attacks import SimBA
from bat.apis.deepapi import VGG16Cifar10

# Load Image [0.0, 1.0]
x = np.asarray(Image.open('dog.jpg').resize((32, 32))) / 255.0

# Initialize the Cloud API Model
DEEP_API_URL = 'https://api.wuhanstudio.uk'
model = VGG16Cifar10(DEEP_API_URL + "/vgg16_cifar10")

# SimBA Attack
simba = SimBA(model)
x_adv = simba.attack(x, epsilon=0.1, max_it=1000)

# Distributed SimBA Attack
x_adv = simba.attack(x, epsilon=0.1, max_it=1000, distributed=True , batch=50, max_workers=10)

```

