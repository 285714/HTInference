include("ht-inf.jl")
include("mixtures.jl")
using LinearAlgebra
using Statistics
using Random


n = 3
k = 2

m = Mixtures.random(n, k)
display(m)

# m.trans .*= 10

L = 1 * transpose(I - m.trans[:, :, 1])
K = -L
Linv = pinv(L)

H_true = HtInf.hitting_times(Linv)
println("\n\nH_true:")
display(H_true)

trails, _ = Mixtures.sample_trails_ct(m, n_trails=10000, trail_len=50)
H_sample = HtInf.hitting_times_ct2(n, trails)
println("\n\nH_sample (better):")
display(H_sample)

H_sample = HtInf.hitting_times_ct(n, trails)
println("\n\nH_sample:")
display(H_sample)

"""
# println("\n\ntrails:")
# display(trails)

println("\n\nlearned:")
display(HtInf.ht_learn(H_true))
"""

m_ = HtInf.em_ct(n, k, trails, num_iters=100, verbose=true)

println("\n\nTrue")
display(m)


