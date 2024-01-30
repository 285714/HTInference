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
    S = np.array([np.ones(n)])
    Ms = np.array([M])
    mixture = dt.Mixture(S, Ms)
    mixture.normalize()
    return mixture

def get_trails(mixture, n_trails, trail_len):
    m = ht.to_mixture([s for s in mixture.S], [M for M in mixture.Ms])
    return ht.get_trails(m, n_trails, trail_len)
    # return ht.get_trails(mixture.S[0], mixture.Ms[0], n_trails, trail_len)

def get_trails_ct(mixture, n_trails, trail_len):
    m = ht.to_mixture([s for s in mixture.S], [K for K in mixture.Ks])
    return ht.get_trails_ct(m, n_trails, trail_len)

def em(n, k, trails, num_iters=100):
    m = ht.em(n, k, trails, num_iters=num_iters)
    S, Ms = ht.from_mixture(m)
    return dt.Mixture(np.array(S), np.array(Ms))

def em_ct(n, k, trails, num_iters=100):
    m = ht.em_ct(n, k, trails, num_iters=num_iters)
    S, Ms = ht.from_mixture(m)
    return dt.Mixture(np.array(S), np.array(Ms))

