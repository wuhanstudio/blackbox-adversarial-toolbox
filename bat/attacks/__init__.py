r'''
This module implements several black-box attacks against Deep Learning models.

Currently Supported Attacks:

- [SimpleBA](https://arxiv.org/abs/1905.07121)

'''
from bat.attacks.base_attack import BaseAttack

from bat.attacks.simba_attack import SimBA
from bat.attacks.square_attack import SquareAttack
from bat.attacks.bandits_attack import BanditsAttack
