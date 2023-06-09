r'''
Black-box Adversarial Toolbox (BAT) is a python library for **Distrubuted Black-box Attacks** against Deep Learning Cloud Services.


## Installation

```
pip install blackbox-adversarial-toolbox
```

Then you can use the cli tool `bat` to try distributed black-box attacks.

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

<br />

## bat.apis

Use `bat api list` to list supported Cloud APIs:

```
1 : deepapi     An open-source image classification cloud service for research on black-box adversarial attacks.
2 : google      Google Cloud Vision AI.
3 : imagga      Imagga automatic tagging API.
```

<br />

## bat.attacks

Use `bat attack list` to list supported attacks:

```
1 : SimBA               Local Search
2 : Square Attack       Local Search
3 : Bandits Atack       Gradient Estimation
```

Local Search:

- SimBA Attack [(Guo et al., 2019)](https://arxiv.org/abs/1905.07121)
- Square Attack [(Andriushchenko et al., 2020)](https://arxiv.org/abs/1912.00049)

Gradient Estimation:

- Bandits Attack [(Ilyas et al., 2019)](https://arxiv.org/abs/1807.07978)

Distributed Black-box Attacks:

- DeepAPI [(Wu et al., 2023)](https://arxiv.org/abs/2210.16371)

<br />

## bat.examples

Use `bat example list` to list available examples:

```
1 : simba_deepapi       SimBA Attack against DeepAPI
2 : bandits_deepapi     Bandits Attack against DeepAPI
3 : square_deepapi      Square Attack against DeepAPI
```

Use `bat example run` to run examples:

```
Usage: bat example run [OPTIONS] COMMAND [ARGS]...

  Run examples

Options:
  --help  Show this message and exit.

Commands:
  bandits_deepapi  Bandits Attack against DeepAPI
  simba_deepapi    SimBA Attack against DeepAPI
  square_deepapi   Square Attack against DeepAPI
```

For example, the command `bat example run simba_deepapi` initiates a distributed SimBA attack against DeepAPI.

<br />

## bat.utils

This module implements utility functions.

'''

# Project Imports
from bat import apis
from bat import attacks
from bat import examples
from bat import utils

# Semantic Version
__version__ = "0.1.2"
