print("compiling julia code ...")
import numpy as np
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main
Main.include("Julia/ht-inf.jl")
print("... done")

ht = Main.HtInf


import dtlearn as dt
import ctlearn as ct


def compute_Linv_ct(mixture):
    K = mixture.Ks[0]
    return np.linalg.pinv(-K.T)

def hitting_times_ct(mixture):
    Linv = compute_Linv_ct(mixture)
    return ht.hitting_times(Linv)

def compute_Linv(mixture):
    M = mixture.Ms[0]
    n, _ = M.shape
    return np.linalg.pinv(np.eye(n) - M.T)

def hitting_times(mixture):
    Linv = compute_Linv(mixture)
    return ht.hitting_times(Linv)

def hitting_times_from_trails(n, trails):
    return ht.hitting_times(n, trails)

def hitting_times_from_trails_ct(n, trails):
    return ht.hitting_times_ct(n, trails)

def ht_learn(H, max_iter=10000, return_loss=False, return_time=False):
    n, _ = H.shape
    M = ht.ht_learn(H, max_iter=max_iter, return_loss=return_loss, return_time=return_time)

    loss, time = None, None
    if return_loss and return_time:
        M, loss, time = M
    elif return_loss:
        M, loss = M
    elif return_time:
        M, time = M

    S = np.array([np.ones(n)])
    Ms = np.array([M])
    mixture = dt.Mixture(S, Ms)
    mixture.normalize()

    if return_loss and return_time:
        return mixture, loss, time
    elif return_loss:
        return mixture, loss
    elif return_time:
        return mixture, time
    else:
        return mixture

def get_trails(mixture, n_trails, trail_len):
    m = ht.to_mixture([s for s in mixture.S], [M for M in mixture.Ms])
    return ht.get_trails(m, n_trails, trail_len)
    # return ht.get_trails(mixture.S[0], mixture.Ms[0], n_trails, trail_len)

def get_trails_ct(mixture, n_trails, trail_len):
    m = ht.to_mixture([s for s in mixture.S], [M for M in mixture.Ms])
    return ht.get_trails_ct(m, n_trails, trail_len)

def em(n, k, trails, num_iters=100):
    m = ht.em(n, k, trails, num_iters=num_iters)
    S, Ms = ht.from_mixture(m)
    return dt.Mixture(np.array(S), np.array(Ms))

def em_ct(n, k, trails, num_iters=100, learn_start=True):
    kwargs = {} if learn_start else {"learn_start": False}
    m = ht.em_ct(n, k, trails, num_iters=num_iters, **kwargs)
    S, Ms = ht.from_mixture(m)
    return ct.Mixture(np.array(S), np.array([M - np.eye(n) for M in Ms]))

def time_htlearn_naive(H, max_iter=10000):
    return ht.time_htlearn_naive(H, max_iter=max_iter)

