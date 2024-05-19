import numpy as np
import networkx as nx

import dtlearn as dt
import ctlearn as ct



def lstsq(_, b, Ainv):
    return Ainv @ b

def grad_loss(Linv, L, H_train, weights=None):
    n, _ = Linv.shape
    if weights is None: weights = np.ones((n, n))

    rowsum_L = lstsq(Linv, np.ones(n), L)
    p = Linv @ rowsum_L
    d = 1 - p
    norm_d = np.linalg.norm(d, 1)
    s = d / norm_d

    colsum = np.sum(Linv, axis=0)
    A = colsum[:, None] - colsum[None, :]
    LinvD = Linv.T - np.diag(Linv)
    B = LinvD / s
    H = A - B

    colsum_P = lstsq(Linv.T, colsum.T, L.T)
    H_diff = weights * (H - H_train)
    H_diff_s = H_diff / s[None, :]
    rowsum_H_diff = np.sum(H_diff, axis=1)
    colsum_H_diff = np.sum(H_diff, axis=0)
    f = 1 - 1 / s
    yi1 = f * colsum_H_diff
    yi2 = rowsum_H_diff - np.diag(H_diff_s)
    yi = yi1 - yi2
    yj = rowsum_H_diff - colsum_H_diff
    x = np.sum(H_diff_s * B, axis=0)
    Jx = (s @ x - x) / norm_d
    LinvJx = Linv.T @ Jx
    LinvJxL = lstsq(Linv.T, LinvJx.T, L.T)
    LinvJxLsq = lstsq(Linv, LinvJxL, L)

    dx1 = (1 - colsum_P[:, None]) * (LinvJxLsq[None, :] - LinvJxLsq[:, None])
    ds = dx1 + (rowsum_L[None, :] - rowsum_L) * (Jx - LinvJxL)

    dloss_part = yj[None, :] + yi[:, None] - H_diff_s.T
    dloss = dloss_part + ds
    dloss[range(n), range(n)] = 0

    return np.linalg.norm(H_diff)**2 / 2, dloss


def ht_from_Linv(Linv, L):
    s = stationary(Linv, L)
    colsum = np.sum(Linv, axis=0)
    A = colsum[:, None] - colsum[None, :]
    B = (Linv.T - np.diag(Linv)) / s[None, :]
    H = A - B
    return H


def stationary(Linv, L):
    n, _ = Linv.shape
    p = Linv @ np.sum(L, axis=1)
    d = 1 - p
    return d / np.sum(d)


def ht_from_trails(n, trails, weights=None, return_var=False):
    assert(weights is None or not return_var)
    if weights is None: weights = np.ones(len(trails))
    ht_sum = np.zeros((n, n))
    ht_cnt = np.zeros((n, n)) # np.ones((n, n)) * 1e-10
    ht_all = [[[] for y in range(n)] for x in range(n)]

    for trail, weight in zip(trails, weights):
        positions = [[] for _ in range(n)]
        for i, x in enumerate(trail):
            positions[x].append(i)

        indices = np.zeros(n, dtype=int)
        for i, x in enumerate(trail):
            for y in range(n):
                if indices[y] < len(positions[y]):
                    ht = positions[y][indices[y]] - i
                    ht_sum[x, y] += weight * ht
                    ht_cnt[x, y] += weight
                    if return_var: ht_all[x][y].append(ht)
            indices[x] += 1

    ret = (ht_sum / ht_cnt, ht_cnt)
    if return_var:
        var = np.array([[np.var(ht, ddof=1) for ht in ht_x] for ht_x in ht_all])
        ret = (*ret, var)
    return ret


def sample_trails(mixture, trail_len: int, n_trails: int):
    k, n = mixture.S.shape
    trails = []
    chains = []
    for i in range(n_trails):
        l, x = divmod(np.random.choice(n * k, p=mixture.S.flatten()), n)
        trail = [x]
        trails.append(trail)
        chains.append(l)
        for _ in range(trail_len - 1):
            x = np.random.choice(n, p=mixture.Ms[l, x])
            trail.append(x)

    return trails, chains


def adam(Linv, loss_grad, eta=1e-5, eps=1e-10, beta1=0.0, beta2=0.999, max_iter=100000, loss_threshold=1e-7, verbose=False):
    n, _ = Linv.shape

    loss_threshold_count = 0
    m = np.zeros_like(Linv)
    v = np.zeros_like(Linv)
    L = np.linalg.pinv(Linv)
    min_loss_Linv = Linv
    min_loss = np.inf

    for i in range(max_iter):
        # projection
        Linv -= np.sum(Linv, axis=1)[:, None] / n
        L = np.linalg.pinv(Linv, rcond=1e-3)
        M = np.eye(n) - L.T
        if np.any(M < 0):
            M[M < 0] = 0
            M /= np.sum(M, axis=1)[:, None]
            L = np.eye(n) - M.T
            Linv = np.linalg.pinv(L, rcond=1e-3)

        # L = np.linalg.pinv(Linv, rcond=1e-3)
        l, g = loss_grad(Linv, L)

        if l < min_loss:
            min_loss = l
            min_loss_Linv = Linv

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dLinv = eta * m_hat / (np.sqrt(v_hat) + eps)
        Linv -= dLinv

        if l < loss_threshold:
            loss_threshold_count += 1
        else:
            loss_threshold_count = 0

        if loss_threshold_count >= 10:
            break

        if verbose and i % 1000 == 0:
            print(f"[ADAM] Iteration {i}: loss={l:.10f}")

        # import pdb; pdb.set_trace()

    return min_loss_Linv, min_loss


def learn(H_train):
    n, _ = H_train.shape

    M_init = wittmann(H_train)
    Linv_init = np.linalg.pinv(np.eye(n) - M_init.T)
    Linv_init -= np.sum(Linv_init, axis=1)[:, None] / n

    Linv, loss = adam(Linv_init, lambda Linv, L: grad_loss(Linv, L, H_train), verbose=True, max_iter=100000)
    M = np.eye(n) - np.linalg.pinv(Linv, rcond=1e-3).T
    return M, loss


def wittmann(H):
    n, _ = H.shape
    M = np.zeros_like(H)
    for x in range(n):
        Hx = np.copy(H).T
        Hx[x, :] = 1
        hx = H[x, :] - 1
        hx[x] = 1
        px, _, _, _ = np.linalg.lstsq(Hx, hx, rcond=None)
        M[x, :] = px
    return M


def rnd_stoch(n):
    M = np.random.random((n, n))
    M = M / np.sum(M, axis=1)[:, None]
    return M


"""
n = 5

mixture = dt.Mixture.random(n, 1)
mixture.S[1:] *= 0.1

# graph = nx.lollipop_graph(3, 2)
# mixture.Ms[0] = nx.to_numpy_array(graph)

mixture.normalize()

M = mixture.Ms[0]
# M = np.array([[0.2, 0.5, 0.3], [0.2, 0.4, 0.4], [0.8, 0.1, 0.1]])
L = np.eye(n) - M.T
Linv = np.linalg.pinv(L)

H_true = ht_from_Linv(Linv, L)
print(M)
print(H_true)
H_max = np.max(H_true)
print(H_max)

trails, _ = sample_trails(mixture, trail_len=10, n_trails=1000)
H_train, cnt, var = ht_from_trails(n, trails, return_var=True)


# x, y = 1, 2
# H_train[x, y] = 1000
# weights = np.ones_like(H_train)
# weights[x, y] = 0

print(cnt)
print(var)
weights = None
# weights = cnt > 2000
# weights = cnt / var
# weights[range(n), range(n)] = 0
# print(weights)

# H_train = H1
# H_train = np.array([[0, 2.5, 3], [2.5, 0, 2.5], [1.5, 3.4, 0]])

M_learned = wittmann(H_train)
print(M_learned)
print("(wittmann) error:", np.sum(np.abs(M - M_learned)))
Linv_wittmann = np.linalg.pinv(np.eye(n) - M_learned.T)
Linv_wittmann -= np.sum(Linv_wittmann, axis=1)[:, None] / n
print(Linv_wittmann)

Linv_start = np.copy(Linv_wittmann)
# Linv_start = -np.ones((n, n)) / (n - 1)
# Linv_start[range(n), range(n)] = 1
# Linv_start = np.linalg.pinv(np.eye(n) - rnd_stoch(n).T)

Linv_learned = adam(Linv_start, lambda Linv, L: grad_loss(Linv, L, H_train, weights=weights), verbose=True)
M_learned2 = np.eye(n) - np.linalg.pinv(Linv_learned, rcond=1e-3).T
H_learned =  ht_from_Linv(Linv_learned, np.linalg.pinv(Linv_learned, rcond=1e-3))
print(M_learned2)

print(f"(gd) error:", np.sum(np.abs(M - M_learned2)), np.linalg.norm(M - M_learned2)**2 / 2)
"""


