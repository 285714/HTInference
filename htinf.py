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

def ht_learn(H):
    n, _ = H.shape
    M = ht.ht_learn(H)
    S = np.array([np.ones(n)])
    Ms = np.array([M])
    mixture = dt.Mixture(S, Ms)
    mixture.normalize()
    return mixture

def get_trails(mixture, n_trails, trail_len):
    m = ht.to_mixture([mixture.S[0]], [mixture.Ms[0]])
    return ht.get_trails(m, n_trails, trail_len)
    # return ht.get_trails(mixture.S[0], mixture.Ms[0], n_trails, trail_len)


