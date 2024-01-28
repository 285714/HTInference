import numpy as np
print("loading")
from julia.api import Julia
jl = Julia(compiled_modules=False)

"""
from julia import Base
print("---->")
print(Base.sind(90))
print(Base.sind(0))
"""

from julia import Main
Main.include("ht-inf.jl")

n = 10
Linv = np.random.random((n, n))
Linv /= np.sum(Linv, 0)[:, None]
print(Main.HtInf.hitting_times(Linv))


# from julia import LinearAlgebra

