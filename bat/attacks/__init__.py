r'''
This module implements Distributed Black-box Attacks against Deep Learning models.

Local Search:

- SimBA Attack [(Guo et al., 2019)](https://arxiv.org/abs/1905.07121)
- Square Attack [(Andriushchenko et al., 2020)](https://arxiv.org/abs/1912.00049)

Gradient Estimation:

- Bandits Attack [(Ilyas et al., 2019)](https://arxiv.org/abs/1807.07978)

#### bat.attacks.simba_attack
#### bat.attacks.square_attack
#### bat.attacks.bandits_attack
'''

from bat.attacks.base_attack import BaseAttack

from bat.attacks.simba_attack import SimBA
from bat.attacks.square_attack import SquareAttack
from bat.attacks.bandits_attack import BanditsAttack
