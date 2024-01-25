import numpy as np
import itertools


def rnd_stoch(n):
    M = np.random.random((n, n))
    M = M / np.sum(M, axis=1)[:, None]
    return M

def rnd_euler(n):
    M = np.zeros((n, n))
    for _ in range(100):
        perm = np.random.permutation(n)
        P = np.zeros((n, n))
        P[range(n), perm] = 1
        M += np.random.random() * P
    M = M / np.sum(M, axis=1)[:, None]
    return M

def rnd_sym_stoch(n):
    M = np.zeros((n, n))
    num_perm = int(100 * n * np.log(n))
    weights = np.random.random(num_perm)
    weights /= np.sum(weights)
    for w in weights:
        perm = np.random.permutation(range(n))
        P = np.zeros((n, n))
        P[range(n), perm] = 1
        M += w * P
    """
    M = np.random.random((n, n))
    while np.any(np.sum(M, axis=0) != 1) or np.any(np.sum(M, axis=1) != 1):
        M = M / np.sum(M, axis=0)[None, :]
        M = M / np.sum(M, axis=1)[:, None]
    """
    return (M + M.T) / 2

def hitting_times_sample(M):
    n = len(M)
    H = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            ls = []
            for _ in range(1000):
                z = x
                l = 0
                while z != y:
                    z = np.random.choice(range(n), p=M[z, :])
                    l += 1
                ls.append(l)
            H[x, y] = np.mean(ls)
    return H

def stationary_sample(M):
    n = len(M)
    counts = np.zeros(n)
    z = 0
    for _ in range(100000):
        z = np.random.choice(range(n), p=M[z, :])
        counts[z] +=1 
    return counts / np.sum(counts)

def stationary(M):
    n = len(M)
    L = np.eye(n) - M.T

    A = np.vstack((L, np.ones(n)))
    b = np.hstack((np.zeros(n), 1))
    s, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # or: np.linalg.solve(L + 1, np.ones(n))

    return s

def hitting_times(M):
    n = len(M)
    L = np.eye(n) - M.T
    Linv = np.linalg.pinv(L)

    return hitting_times_Linv(Linv)

    """
    A = np.vstack((L, np.ones(n)))
    b = np.hstack((np.zeros(n), 1))
    s, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    X = np.eye(n)[:, None, :] - np.eye(n)[None, :, :]
    S = 1 - np.eye(n) / s[:, None]
    H = np.einsum("ib,ij,abj->ab", S, Linv, X)
    """

    """
    base = np.eye(n)
    H = np.zeros((n, n))
    for u, v in itertools.product(range(n), repeat=2):
        H[u, v] = (1 - base[v] / s[v]).T @ Linv @ (base[u] - base[v])
    """

    return H

def hitting_times_Linv(Linv):
    # M = np.eye(n) - np.linalg.pinv(Linv, rcond=1e-10).T
    # s = stationary(M)
    s = stationary_Linv(Linv)
    return hitting_times_Linv_sfix(Linv, s)

def hitting_times_Linv_sfix(Linv, s):
    X = np.eye(n)[:, None, :] - np.eye(n)[None, :, :]
    S = 1 - np.eye(n) / s[:, None]
    H = np.einsum("ib,ij,abj->ab", S, Linv, X)
    return H

def hitting_times_loss(H_true, Linv):
    return np.linalg.norm(H_true - hitting_times_Linv(Linv))**2 / 2

def hitting_times_loss_sfix(H_true, Linv, s):
    return np.linalg.norm(H_true - hitting_times_Linv_sfix(Linv, s))**2 / 2

def rk1_update(Linv, u, v, eps=0.1):
    Linv = np.copy(Linv)
    Linv[u, v] += eps
    Linv[u, u] -= eps
    return Linv

def numeric_grad(loss, Linv, eps=1e-12):
    dLinv = np.zeros_like(Linv)
    loss0 = loss(Linv)
    d = np.zeros_like(Linv)
    for u, v in itertools.product(range(n), repeat=2):
        d[u, v] += eps
        d[u, u] -= eps
        dLinv[u, v] = loss(Linv + d) - loss0
        d[u, v] -= eps
        d[u, u] += eps
    return dLinv / eps

def grad_Linv_sfix(H_true, Linv, s):
    # loss_grad = H_true - hitting_times_Linv_sfix(Linv, s)
    H = hitting_times_Linv_sfix(Linv, s)
    dLinv = np.zeros_like(Linv)
    for i, j in itertools.product(range(n), repeat=2):
        for u, v in itertools.product(range(n), repeat=2):
            dLinv[i, j] += (1 - (v == i) / s[v]) * ((u == j) - (v == j)) * (H[u, v] - H_true[u, v])
    return dLinv

def stationary_Linv2(Linv):
    p, _, _, _ = np.linalg.lstsq(Linv, np.ones(n), rcond=None)
    v = 1 - Linv @ p
    s = v / np.sum(v)
    return s

def stationary_Linv(Linv):
    L = np.linalg.pinv(Linv, rcond=1e-10)
    p = Linv @ L @ np.ones(n)
    # p, _, _, _ = np.linalg.lstsq(Linv, np.ones(n), rcond=None)
    v = 1 - Linv @ p
    s = v / np.sum(v)
    return s

def sm_update_one(L):
    Linv = np.linalg.pinv(L)
    X1 = np.linalg.inv(L + 1)
    s1 = X1 @ np.ones(n)

    h = np.sum(Linv, axis=0)
    v = np.sum(np.eye(n) - Linv @ L, axis=0)
    v2 = 1 - Linv @ L @ np.ones(n)
    v3 = 1 - Linv @ np.linalg.lstsq(Linv, np.ones(n), rcond=None)[0]
    uinv = np.ones(n) / n
    vinv = v / np.linalg.norm(v)**2

    X2 = Linv + np.outer(vinv, uinv - h)
    s2 = X2 @ np.ones(n)
    s3 = vinv
    s4 = v / np.sum(v)

    import pdb; pdb.set_trace()

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
    X5 = Ainv + 1/beta * Ainv @ np.outer(h, u) - beta/sigma2 * np.outer(p2, q2)

    # np.outer(v, h) gives the first row of Ainv - X1

    import pdb; pdb.set_trace()

def sm_update_(A, c, d):
    m, n = A.shape
    Ainv = np.linalg.pinv(A)
    X1 = np.linalg.pinv(A + np.outer(c, d))

    k = Ainv @ c
    h = d @ Ainv
    kinv = k / np.linalg.norm(k)**2
    hinv = h / np.linalg.norm(h)**2
    u = (np.eye(m) - A @ A.T) @ c
    v = d @ (np.eye(n) - A.T @ A)
    uinv = u / np.linalg.norm(u)**2
    vinv = v / np.linalg.norm(v)**2
    beta = 1 + d @ Ainv @ c

    X2 = Ainv - np.outer(k, uinv) - np.outer(vinv, h) + beta * np.outer(vinv, uinv)
    X3 = Ainv - Ainv @ np.outer(hinv, h) - np.outer(k, uinv)

    # np.outer(v, h) gives the first row of Ainv - X1

    import pdb; pdb.set_trace()

def adam(X, grad, loss, eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=2000, loss_threshold=1e-20, verbose=False):
    loss_threshold_count = 0
    m = np.zeros_like(X)
    v = np.zeros_like(X)

    for i in range(max_iter):
        g = grad(X)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dX = eta * m_hat / (np.sqrt(v_hat) + eps)
        X -= dX

        # projection:

        # eigvals = np.linalg.eigvals(X)
        # assert(np.imag(np.min(eigvals)) < 1e-20)
        # X -= np.real(np.min(eigvals)) * np.eye(n)

        # X[range(n), range(n)] -= np.sum(X, axis=1)

        d = np.sum(X, axis=1)
        X -= d[:, None] / n

        l = loss(X)
        if l < loss_threshold: loss_threshold_count += 1
        else: loss_threshold_count = 0

        if loss_threshold_count >= 10: break

        if verbose and i % 100 == 0:
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                print(f"Iteration {i:5d}: loss={l:.20f}")
                # print("     pre:", eigvals)
                # print("    post:", np.sort(np.linalg.eigvals(X)))
                # print("    true:", np.sort(np.linalg.eigvals(Linv_true)))

    return X



# rnd =  rnd_stoch
# rnd = rnd_euler
rnd = rnd_sym_stoch


n = 5
# M_true = rnd_sym_stoch(n)
M_true = rnd(n)

H_true = hitting_times(M_true)
s_true = stationary(M_true)
L_true = np.eye(n) - M_true.T
Linv_true = np.linalg.pinv(L_true)


# M = rnd_sym_stoch(n)
M = rnd(n)
L = np.eye(n) - M.T
Linv = np.linalg.pinv(L)


# test rank-1 update to L:
# X = L
# X = np.random.random((n, n))
# X = np.vstack((np.zeros(n), np.random.random((n, n-1)).T)).T
# A = np.vstack((X, np.zeros(n)))
# c = np.hstack((np.zeros(n), 1))
# d = np.ones(n)
# sm_update(A, c, d)
# sm_update(L, np.ones(n), np.ones(n))
# sm_update_one(L)


# loss = lambda Linv: hitting_times_loss_sfix(H_true, Linv, s_true)
# grad = lambda Linv: grad_Linv_sfix(H_true, Linv, s_true)

loss = lambda Linv: hitting_times_loss(H_true, Linv)
grad = lambda Linv: numeric_grad(loss, Linv)

Linv = adam(Linv, grad, loss, verbose=True)

"""
eta = 1e-5
for i in range(10000):
    dLinv = grad(Linv)
    Linv -= eta * dLinv

    eigvals = np.linalg.eigvals(Linv)
    Linv -= np.real(np.min(eigvals)) * np.eye(n)

    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss(Linv)}, min eigval before projecting={np.min(eigvals)}")
"""

# print(Linv)
L = np.linalg.pinv(Linv, rcond=1e-10)
M = np.eye(n) - L.T
# M /= np.sum(M, axis=1)[:, None]

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
print("\n   Learned H (through M)")
print(hitting_times(M))
print("\n   Learned H (through Linv)")
print(hitting_times_Linv(Linv))




