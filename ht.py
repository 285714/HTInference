import numpy as np
import itertools
import rnd
import time


lstsq_time = 0
def stationary(Linv, L=None):
    global lstsq_time
    n = len(Linv)
    if L is None:
        start = time.time()
        p, _, _, _ = np.linalg.lstsq(Linv, np.ones(n), rcond=1e-10)
        lstsq_time += time.time() - start
    else:
        p = Linv @ L @ np.ones(n)
    v = 1 - Linv @ p
    s = v / np.sum(v)
    return s

def hitting_times(Linv, L=None):
    s = stationary(Linv, L=L)
    return hitting_times_sfix(Linv, s)

def hitting_times_sfix(Linv, s):
    n = len(Linv)

    X = np.eye(n)[:, None, :] - np.eye(n)[None, :, :]
    S = 1 - np.eye(n) / s[:, None]
    H = np.einsum("ib,ij,abj->ab", S, Linv, X)

    """
    base = np.eye(n)
    H = np.zeros((n, n))
    for u, v in itertools.product(range(n), repeat=2):
        H[u, v] = (1 - base[v] / s[v]) @ Linv @ (base[u] - base[v])
    """

    return H

def hitting_times_loss_sfix(H_true, Linv, s):
    return np.linalg.norm(H_true - hitting_times_sfix(Linv, s))**2 / 2


def l2loss(H_true, H):
    return np.linalg.norm(H - H_true)**2 / 2

def grad_l2loss(H_true, H):
    return H - H_true

def grad_H_sfix(Linv, s):
    # H:     u x v
    # Linv:  i x j
    # dLinv: (u x v) x (i x j)
    n = len(Linv)
    X = 1 - np.diag(1 / s)
    B = np.eye(n)[:, None] - np.eye(n)[None, :]
    dH = np.einsum("vi,jix,uvx->ijuv", X, B, B)

    """
    base = np.eye(n)
    dH = np.zeros((n, n, n, n))
    for u, v, i, j in itertools.product(range(n), repeat=4):
        dLinv = np.outer(base[i], base[j] - base[i])
        dH[i, j, u, v] = (1 - base[v] / s[v]) @ dLinv @ (base[u] - base[v])
    """

    return dH

def grad_s(Linv, L=None):
    return numeric_grad(stationary, Linv)
    """
    n = len(Linv)
    s0 = stationary(Linv)
    d = np.zeros_like(Linv)
    ds = np.zeros((n, n, n))
    for u, v in itertools.product(range(n), repeat=2):
        d[u, v] += eps
        d[u, u] -= eps
        ds[u, v] = stationary(Linv + d) - s0
        d[u, v] -= eps
        d[u, u] += eps
    return ds / eps
    """

def grad_H(Linv, L=None):
    n = len(Linv)
    base = np.eye(n)
    s = stationary(Linv, L=L)
    dH1 = grad_H_sfix(Linv, s) # np.zeros((n, n, n, n))
    dH2 = np.zeros((n, n, n, n))
    ds = grad_s(Linv, L=L)
    X = np.diag(1 / s**2)
    B = np.eye(n)[:, None] - np.eye(n)[None, :]

    """
    for u, v, i, j in itertools.product(range(n), repeat=4):
        # dLinv = np.outer(base[i], base[j] - base[i])
        # d1 = (1 - base[v] / s[v]) @ dLinv @ (base[u] - base[v])
        d2 = X[v] * ds[i, j, v] @ Linv @ B[u, v]
        dH2[i, j, u, v] = d2
    """

    dH2 = np.einsum("va,ijv,ab,uvb->ijuv", X, ds, Linv, B)

    """
    for u, v, i, j in itertools.product(range(n), repeat=4):
        # dLinv = np.outer(base[i], base[j] - base[i])
        # d1 = (1 - base[v] / s[v]) @ dLinv @ (base[u] - base[v])
        d2 = base[v] / s[v]**2 * ds[i, j, v] @ Linv @ (base[u] - base[v])
        dH2[i, j, u, v] = d2
    """

    # dH_ = numeric_grad(hitting_times, Linv)
    # dH_1 = numeric_grad(lambda Linv: hitting_times_sfix(Linv, s), Linv)
    # dH_2 = grad_H_sfix(Linv, s)
    # import pdb; pdb.set_trace()
    return dH1 + dH2

def grad_loss_sfix(Linv, s, H_true):
    H = hitting_times_sfix(Linv, s)
    dH = grad_H_sfix(Linv, s)
    dloss = grad_l2loss(H_true, H)
    grad = np.tensordot(dH, dloss)
    grad[range(n), range(n)] = -np.sum(grad, axis=1)
    return grad

def grad_loss(Linv, H_true, L=None):
    H = hitting_times(Linv, L=L)
    dH = grad_H(Linv, L=L)
    dloss = grad_l2loss(H_true, H)
    grad = np.tensordot(dloss, dH)
    grad[range(n), range(n)] = -np.sum(grad, axis=1)
    return grad

def loss_sfix(Linv, s, H_true):
    H = hitting_times_sfix(Linv, s)
    return l2loss(H_true, H)

def loss(Linv, H_true, L=None):
    H = hitting_times(Linv, L=L)
    return l2loss(H_true, H)


def numeric_grad(loss, Linv, eps=1e-12):
    loss0 = np.array(loss(Linv))
    dLinv = np.zeros(Linv.shape + loss0.shape)
    d = np.zeros_like(Linv)
    for u, v in itertools.product(range(n), repeat=2):
        d[u, v] += eps
        d[u, u] -= eps
        dLinv[u, v] = loss(Linv + d) - loss0
        d[u, v] -= eps
        d[u, u] += eps
    return dLinv / eps



def adam(Linv, grad, loss, eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=20000, loss_threshold=1e-20, verbose=False):
    loss_threshold_count = 0
    m = np.zeros_like(Linv)
    v = np.zeros_like(Linv)

    for i in range(max_iter):
        L = np.linalg.pinv(Linv, rcond=1e-10)
        g = grad(Linv, L=L)
        l = loss(Linv, L=L)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dX = eta * m_hat / (np.sqrt(v_hat) + eps)
        Linv -= dX

        d = np.sum(Linv, axis=1)
        Linv -= d[:, None] / n

        if l < loss_threshold: loss_threshold_count += 1
        else: loss_threshold_count = 0

        if loss_threshold_count >= 10: break

        if verbose and i % 100 == 0:
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                print(f"Iteration {i:5d}: loss={l:.20f}")

    return Linv


def adam_sfix(Linv, grad, loss, eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=20000, loss_threshold=1e-20, verbose=False):
    loss_threshold_count = 0
    m = np.zeros_like(Linv)
    v = np.zeros_like(Linv)

    for i in range(max_iter):
        s = stationary(Linv)
        g = grad(Linv, s)
        l = loss(Linv, s)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dX = eta * m_hat / (np.sqrt(v_hat) + eps)
        Linv -= dX

        d = np.sum(Linv, axis=1)
        Linv -= d[:, None] / n

        if l < loss_threshold: loss_threshold_count += 1
        else: loss_threshold_count = 0

        if loss_threshold_count >= 10: break

        if verbose and i % 100 == 0:
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                print(f"Iteration {i:5d}: loss={l:.20f}")

    return Linv



def sm_update(A, c, d):
    m, n = A.shape
    Ainv = np.linalg.pinv(A)
    X1 = np.linalg.pinv(A + np.outer(c, d))

    k = Ainv @ c
    h = d @ Ainv
    kinv = k / np.linalg.norm(k)**2
    hinv = h / np.linalg.norm(h)**2
    u = (np.eye(m) - A @ Ainv) @ c
    v = d @ (np.eye(n) - Ainv @ A)
    uinv = u / np.linalg.norm(u)**2
    vinv = v / np.linalg.norm(v)**2
    beta = 1 + d @ Ainv @ c
    p1 = - np.linalg.norm(k)**2 / beta * v - k
    p2 = - np.linalg.norm(u)**2 / beta * Ainv @ h - k
    q1 = - np.linalg.norm(v)**2 / beta * k @ Ainv - h
    q2 = - np.linalg.norm(h)**2 / beta * u - h
    sigma1 = np.linalg.norm(k)**2 * np.linalg.norm(v)**2 + beta**2
    sigma2 = np.linalg.norm(h)**2 * np.linalg.norm(u)**2 + beta**2

    X2 = Ainv - np.outer(k, uinv) - np.outer(vinv, h) + beta * np.outer(vinv, uinv)
    X3 = Ainv - Ainv @ np.outer(hinv, h) - np.outer(k, uinv)
    X4 = Ainv + 1/beta * np.outer(v, k) @ Ainv - beta/sigma1 * np.outer(p1, q1)
    X5 = Ainv + 1/beta * Ainv @ np.outer(h, u) - beta/sigma2 * np.outer(p2, q2) # <- this one

    import pdb; pdb.set_trace()



n = 10
M_true = rnd.stoch(n)
M_true = np.array([
    [0.03600424950260535, 0.1612773856745104, 0.16231244819018864, 0.050345550174747115, 0.05758500734391162, 0.09911145278742983, 0.12683576433014782, 0.09013208612522552, 0.05140799419246925, 0.1649880616787643],
    [0.1279216244162943, 0.30374074251918487, 0.24641509125830918, 0.017981889853653826, 0.050503965917088556, 0.00698280878261917, 0.05609587207776268, 0.10994704801022592, 0.04315826894021233, 0.0372526882246492],
    [0.17954539132196345, 0.010594753397221924, 0.009200477149307532, 0.07803120380306171, 0.2541835583749725, 0.16858974491696338, 0.02777421049981009, 0.0535726843191172, 0.14947451471188689, 0.06903346150569531],
    [0.18043134765857827, 0.060512024411273374, 0.22631856682781998, 0.03197308350154014, 0.057190552631094634, 0.13852308963277166, 0.004719405721804988, 0.05106901957674922, 0.011452730894726416, 0.23781017914364128],
    [0.024555674686795825, 0.13790664569969252, 0.009289718153472595, 0.10683466374453904, 0.0472455526933794, 0.16096927319292975, 0.11425472149765728, 0.20329712250965679, 0.08530838973803631, 0.11033823808384052],
    [0.20218157946813958, 0.18703695233210216, 0.03000923848302702, 0.044412481973746594, 0.10692992946151846, 0.17776401361900993, 0.1073045329696188, 0.089196584748383, 0.0024006167321941065, 0.052764070212260455],
    [0.1471439063849349, 0.11024969112617113, 0.12791151587898578, 0.11362434717776387, 0.0012929384368293242, 0.07866617296704413, 0.10787822593465805, 0.011770253052851778, 0.09615225653015189, 0.2053106925106092],
    [0.019905134918919667, 0.1861511404395247, 0.013333447968214765, 0.15978595818391556, 0.06625459465475148, 0.16455885119389568, 0.02176081925975192, 0.12737454393376973, 0.1421008321872719, 0.09877467725998461],
    [0.02215959414354327, 0.018475970283131558, 0.13147776653585716, 0.07249153421501284, 0.0821157805740173, 0.23935431938010923, 0.1865136923161999, 0.07153448038034316, 0.048487457416698, 0.12738940475508762],
    [0.16314895930723267, 0.058759978664890924, 0.13834792165857024, 0.03688570368949333, 0.15259190512701448, 0.01403034116454585, 0.12078153706050683, 0.13448495255041468, 0.05162176708949055, 0.1293469336878405]
    ])

M = np.array([
    [0.13600424950260535, 0.0612773856745104, 0.16231244819018864, 0.050345550174747115, 0.05758500734391162, 0.09911145278742983, 0.12683576433014782, 0.09013208612522552, 0.05140799419246925, 0.1649880616787643],
    [0.1279216244162943, 0.30374074251918487, 0.24641509125830918, 0.017981889853653826, 0.050503965917088556, 0.00698280878261917, 0.05609587207776268, 0.10994704801022592, 0.04315826894021233, 0.0372526882246492],
    [0.17954539132196345, 0.010594753397221924, 0.009200477149307532, 0.07803120380306171, 0.2541835583749725, 0.16858974491696338, 0.02777421049981009, 0.0535726843191172, 0.14947451471188689, 0.06903346150569531],
    [0.18043134765857827, 0.060512024411273374, 0.22631856682781998, 0.03197308350154014, 0.057190552631094634, 0.13852308963277166, 0.004719405721804988, 0.05106901957674922, 0.011452730894726416, 0.23781017914364128],
    [0.024555674686795825, 0.13790664569969252, 0.009289718153472595, 0.10683466374453904, 0.0472455526933794, 0.16096927319292975, 0.11425472149765728, 0.20329712250965679, 0.08530838973803631, 0.11033823808384052],
    [0.20218157946813958, 0.18703695233210216, 0.03000923848302702, 0.044412481973746594, 0.10692992946151846, 0.17776401361900993, 0.1073045329696188, 0.089196584748383, 0.0024006167321941065, 0.052764070212260455],
    [0.1471439063849349, 0.11024969112617113, 0.12791151587898578, 0.11362434717776387, 0.0012929384368293242, 0.07866617296704413, 0.10787822593465805, 0.011770253052851778, 0.09615225653015189, 0.2053106925106092],
    [0.019905134918919667, 0.1861511404395247, 0.013333447968214765, 0.15978595818391556, 0.06625459465475148, 0.16455885119389568, 0.02176081925975192, 0.12737454393376973, 0.1421008321872719, 0.09877467725998461],
    [0.02215959414354327, 0.018475970283131558, 0.13147776653585716, 0.07249153421501284, 0.0821157805740173, 0.23935431938010923, 0.1865136923161999, 0.07153448038034316, 0.048487457416698, 0.12738940475508762],
    [0.16314895930723267, 0.058759978664890924, 0.13834792165857024, 0.03688570368949333, 0.15259190512701448, 0.01403034116454585, 0.12078153706050683, 0.13448495255041468, 0.05162176708949055, 0.1293469336878405]
    ])

L_true = np.eye(n) - M_true.T
Linv_true = np.linalg.pinv(L_true)
H_true = hitting_times(Linv_true)

s_true = stationary(Linv_true)

Linv = np.linalg.pinv(np.eye(n) - M.T)
dH = grad_loss(Linv, H_true)
dH2 = numeric_grad(lambda Linv: loss(Linv, H_true=H_true), Linv)
exit()

M = rnd.stoch(n)
L = np.eye(n) - M.T
Linv = np.linalg.pinv(L)





def adam_loss(Linv, L=None):
    return loss(Linv, H_true, L=L)
def adam_grad(Linv, L=None):
    return grad_loss(Linv, H_true, L=L)

start = time.time()
Linv = adam(Linv, adam_grad, adam_loss, verbose=True)
adam_time = time.time() - start

print("adam_time:", adam_time)
print("lstsq_time:", lstsq_time)

L = np.linalg.pinv(Linv, rcond=1e-10)
M = np.eye(n) - L.T

print("\n   True M")
print(M_true)
print("\n   Learned M")
print(M)
print(f"\nrow sums: {np.sum(M, axis=1)}, frob-dist: {np.linalg.norm(M_true - M)}")
print(f"singular values: {np.linalg.svd(L)[1]}")

print("\n\n   True Linv")
print(Linv_true)
print("\n   Learned Linv")
print(Linv)

print("\n\n   True H")
print(H_true)
print("\n   Learned H (through Linv)")
print(hitting_times(Linv))







# problem not convex if not symmetric -> can we still show convergence properties?
# noise -> what's a good loss?
# mixtures?


