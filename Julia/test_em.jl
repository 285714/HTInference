include("ht-inf.jl")
include("mixtures.jl")


n = 5
k = 2

m = Mixtures.random(n, k)
trails, _ = Mixtures.sample_trails(m, n_trails=1000, trail_len=100)
m_ = HtInf.em(n, k, trails, num_iters=100)

