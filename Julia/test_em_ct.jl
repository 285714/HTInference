include("ht-inf.jl")
include("mixtures.jl")
using LinearAlgebra
using Statistics
using Random


n = 5
k = 1

m = Mixtures.random(n, k)
m.start .= 1 / (n * k)
# display(m)

L = I - m.trans[:, :, 1]
K = -L
Linv = pinv(transpose(L))

H_true1 = HtInf.hitting_times(Linv)
println(maximum(H_true1))

# H_true2 = HtInf.hitting_times(pinv(transpose(I - m.trans[:, :, 2])))
# println(maximum(H_true2))


trails, _ = Mixtures.sample_trails_ct(m, n_trails=5000, trail_len=1000)
m_ = HtInf.em_ct(n, k, trails, num_iters=50, verbose=true, m_true=m)

println("\n\nTrue")
display(m)
println("\n\nLearned")
display(m_)


"""
H_true = HtInf.hitting_times(pinv(I - m.trans[:, :, 1]))
println("\n\nH_true:")
display(H_true)

trails, _ = Mixtures.sample_trails_ct(m, n_trails=1000, trail_len=5)
# println(trails)

H_sample = HtInf.hitting_times_ct2(n, trails)
println("\n\nH_sample (better):")
display(H_sample)

H_sample = HtInf.hitting_times_ct(n, trails)
println("\n\nH_sample:")
display(H_sample)
"""

"""
# println("\n\ntrails:")
# display(trails)

println("\n\nlearned:")
display(HtInf.ht_learn(H_true))
"""

"""
m_ = HtInf.em_ct(n, k, trails, num_iters=100, verbose=true)

println("\n\nTrue")
display(m)
"""

