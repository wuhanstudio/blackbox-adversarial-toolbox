"""
This module implements the black-box attack `SimBA`.
"""

import os
import gc
import numpy as np
from tqdm import tqdm
import concurrent.futures

from bat.attacks.base_attack import BaseAttack

SCALE = 255
PREPROCESS = lambda x: x

def proj_lp(v, xi=0.1, p=2):
    """
    SUPPORTS only p = 2 and p = Inf for now
    """
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten('C')))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

class SimBA(BaseAttack):
    """
    Implementation of the `SimBA` attack. Paper link: https://arxiv.org/abs/1905.07121
    """

    def __init__(self,  classifier):
        """
        Create a class: `SimBA` instance.
        - classifier: model to attack
        """
        super().__init__(classifier)

    def init(self, x, max_it):
        """
        Initialize the attack.
        """

        x_adv = x.copy()
        y_pred = self.classifier.predict(PREPROCESS(x.copy()))

        perm = []
        for xi in x:
            perm.append(np.random.permutation(xi.reshape(-1).shape[0]))
            assert len(perm[-1]) > max_it, 'The maxinum number of iteration should be smaller than the image dimension.'

        return x_adv, y_pred, perm

    def step(self, x_adv, y_pred, perm, index, epsilon):
        """
        Single step for non-distributed attack.
        """

        x_adv_plus = []
        x_adv_minus = []
        x_adv_diff = []
        for i in range(0, len(x_adv)):
            diff = np.zeros(x_adv[i].reshape(-1).shape[0])
            diff[perm[i][index]] = epsilon
            diff = diff.reshape(x_adv[i].shape)
            x_adv_plus.append(np.clip(x_adv[i] + diff, 0, 1 * SCALE))
            x_adv_minus.append(np.clip(x_adv[i] - diff, 0, 1 * SCALE))
            x_adv_diff.append(diff)

        plus = self.classifier.predict(PREPROCESS(x_adv_plus.copy()))
        minus = self.classifier.predict(PREPROCESS(x_adv_minus.copy()))
        
        for i in range(0, len(x_adv)):
            if plus[i][np.argmax(y_pred[i])] < y_pred[i][np.argmax(y_pred[i])]:
                x_adv[i] = x_adv[i] + x_adv_diff[i]
                y_pred[i] = plus[i]
            elif minus[i][np.argmax(y_pred[i])] < y_pred[i][np.argmax(y_pred[i])]:
                x_adv[i] = x_adv[i] - x_adv_diff[i]
                y_pred[i] = minus[i]
            else:
                pass

        return x_adv, y_pred

    def batch(self, x_adv, y_pred, perm, index, epsilon, concurrency):
        """
        Single step for distributed attack.
        """
        noises = []
        for i in range(0, len(x_adv)):
            noises.append(np.zeros(x_adv[i].shape))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(self.step, x_adv, y_pred, perm, index+j, epsilon): j for j in range(0, concurrency)}
            for future in concurrent.futures.as_completed(future_to_url):
                j = future_to_url[future]
                try:
                    x_adv_new, _ = future.result()
                    for i in range(0, len(x_adv)):
                        noises[i] = noises[i] + x_adv_new[i] - x_adv[i]
                except Exception as exc:
                    print('Task %r generated an exception: %s' % (j, exc))
                else:
                    pass

        for i in range(0, len(x_adv)):
            if(np.sum(noises[i]) != 0):
                noises = proj_lp(noises[i], xi = 10)
            x_adv[i] = np.clip(x_adv[i] + noises[i], 0, 1 * SCALE)

        y_adv = self.classifier.predict(PREPROCESS(x_adv.copy())) 

        return x_adv, y_adv

    def attack(self, x, y, epsilon=0.05, max_it=1000, concurrency=1):
        """
        Initiate the attack.

        - x: input data
        - y: input labels
        - epsilon: perturbation on each pixel
        - max_it: number of iterations
        - concurrency: number of concurrent threads
        """

        n_targets = 0
        if type(x) == list:
            n_targets = len(x)
        elif type(x) == np.ndarray:
            n_targets = x.shape[0]
        else:
            raise ValueError('Input type not supported...')

        assert n_targets > 0

        # Initialize attack
        x_adv, y_pred, perm = self.init(x, max_it)

        # Compute number of images correctly classified
        y_pred_classes = np.argmax(y_pred, axis=1)
        correct_classified_mask = (y_pred_classes == y)
        correct_classified = [i for i, v in enumerate(correct_classified_mask) if v]

        # Images to attack
        not_dones_mask = correct_classified_mask.copy()

        print('Clean accuracy: {:.2%}'.format(np.mean(correct_classified_mask)))

        if np.mean(correct_classified_mask) == 0:
            print('No clean examples classified correctly. Aborting...')
            n_queries = np.ones(len(x))  # ones because we have already used 1 query

            mean_nq, mean_nq_ae = np.mean(n_queries), np.mean(n_queries)

            return x

        else:
            if n_targets > 1:
                # Horizontally Distributed Attack
                pbar = tqdm(range(0, max_it), desc="Distributed SimBA Attack (Horizontal)")
            else:
                # Vertically Distributed Attack
                pbar = tqdm(range(0, max_it, concurrency), desc="Distributed SimBA Attack (Vertical)")

        total_queries = np.zeros(len(x))

        for i_iter in pbar:

            not_dones = [i for i, v in enumerate(not_dones_mask) if v]

            x_adv_curr = [x_adv[idx] for idx in not_dones]
            y_curr = [y_pred[idx] for idx in not_dones]
            perm_curr = [perm[idx] for idx in not_dones]

            y_curr = np.array(y_curr)

            if n_targets > 1:
                # Horizontally Distributed Attack
                x_adv_curr, y_curr = self.step(x_adv_curr, y_curr, perm_curr, i_iter, epsilon*SCALE)
            else:
                # Vertically Distributed Attack
                x_adv_curr, y_curr = self.batch(x_adv_curr, y_curr, perm_curr, i_iter, epsilon*SCALE, concurrency)

            for i in range(len(not_dones)):
                x_adv[not_dones[i]] = x_adv_curr[i]
                y_pred[not_dones[i]] = y_curr[i]

            # Logging stuff
            if n_targets > 1:
                # Horizontally Distributed Attack
                total_queries += 2 * not_dones_mask
            else:
                # Vertically Distributed Attack
                total_queries += 2 * concurrency * not_dones_mask + 1

            y_pred_classes = np.argmax(y_pred, axis=1)
            not_dones_mask = not_dones_mask * (y_pred_classes == y)

            success_mask = correct_classified_mask * (1 - not_dones_mask)
            num_success = success_mask.sum()
            current_success_rate = (num_success / correct_classified_mask.sum())

            if num_success == 0:
                success_queries = -1
            else:
                success_queries = ((success_mask * total_queries).sum() / num_success)

            pbar.set_postfix({'Total Queries': total_queries.sum(), 'Mean Higest Prediction': y_pred[correct_classified].max(axis=1).mean(), 'Attack Success Rate': current_success_rate, 'Avg Queries': success_queries})

            acc = not_dones_mask.sum() / correct_classified_mask.sum()
            mean_nq, mean_nq_ae = np.mean(total_queries), np.mean(total_queries *success_mask)

            # Early break
            if current_success_rate == 1.0:
                break

            gc.collect()

        return x_adv
