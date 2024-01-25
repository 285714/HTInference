import numpy as np


def stoch(n):
    M = np.random.random((n, n))
    M = M / np.sum(M, axis=1)[:, None]
    return M

def euler(n):
    M = np.zeros((n, n))
    for _ in range(100):
        perm = np.random.permutation(n)
        P = np.zeros((n, n))
        P[range(n), perm] = 1
        M += np.random.random() * P
    M = M / np.sum(M, axis=1)[:, None]
    return M

def sym_stoch(n):
    M = np.zeros((n, n))
    num_perm = int(100 * n * np.log(n))
    weights = np.random.random(num_perm)
    weights /= np.sum(weights)
    for w in weights:
        perm = np.random.permutation(range(n))
        P = np.zeros((n, n))
        P[range(n), perm] = 1
        M += w * P
    return (M + M.T) / 2
