include("ht-inf.jl")
using LinearAlgebra


n = 100
k = 1

m = HtInf.Mixtures.random(n, k)
# m.start[:] .= 1 / (n * k)

trails, _ = HtInf.Mixtures.sample_trails(m, n_trails=10, trail_len=100)

M = m.trans[:, :, 1]
L = transpose(I - M)
Linv = pinv(L)
H = HtInf.hitting_times(Linv)
println("\n\nTrue HT")
display(H)

# println("\n\ntrails")
# println(trails)

HtInf.time_htlearn_naive(H)

"""
println("\n\nEstimated HT")
H_est = HtInf.hitting_times2(n, trails)
display(H_est)
"""

"""
m_ = HtInf.em(n, k, trails, num_iters=100, verbose=false)

println("\n\nLearned Mixture:")
display(m_)
println("\n\nTrue Mixture:")
display(m)

println("\n\nerror=", HtInf.Mixtures.distance(m, m_))
"""

