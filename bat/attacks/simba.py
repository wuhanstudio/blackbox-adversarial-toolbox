"""
This module implements the black-box attack `SimBA`.
"""

import numpy as np

class SimBA():
    """
    Implementation of the `SimBA` attack. Paper link: https://arxiv.org/abs/1905.07121
    """

    def __init__(self):
        """
        Create a class: `SimBA` instance.
        """

    def attack(self, x, model, epsilon=0.1, epoch=1000):
        """
        Initiate the attack.

        - x: input data
        - model: model to attack
        - epsilon: perturbation on each pixel
        - epoch: number of iterations
        """
        y_pred = model.predict(np.array([x]))[0]

        x_adv = x
        y_list = []
        y_list.append(np.argmax(y_pred))

        perm = np.random.permutation(x.reshape(-1).shape[0])
        for i in range(0, epoch):
            diff = np.zeros(x.reshape(-1).shape[0])
            diff[perm[i]] = epsilon
            diff = diff.reshape(x.shape)

            y_attack = None

            plus = model.predict( np.array([np.clip(x_adv + diff, 0, 1)]) )[0]
            if plus[np.argmax(y_pred)] < y_list[-1]:
                x_adv = np.clip(x_adv + diff, 0, 1)
                y_list.append(plus[np.argmax(y_pred)])
                print(i, plus[np.argmax(y_pred)], np.sqrt(np.power(x_adv - x, 2).sum()))
                y_attack = plus
            else:
                minus = model.predict( np.array([np.clip(x_adv - diff, 0, 1)]) )[0]
                if minus[np.argmax(y_pred)] < y_list[-1]:
                    x_adv = np.clip(x_adv - diff, 0, 1)
                    y_list.append(minus[np.argmax(y_pred)])
                    print(i, minus[np.argmax(y_pred)], np.sqrt(np.power(x_adv - x, 2).sum()))
                y_attack = minus

            # Early break
            if(np.argmax(y_attack) != np.argmax(y_pred)):
                break
        return x_adv    
