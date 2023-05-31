import numpy as np

def proj_lp(v, xi=0.1, p=2):
    """
    Supports only p = 2 and p = Inf.
    """
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten('C')))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v
