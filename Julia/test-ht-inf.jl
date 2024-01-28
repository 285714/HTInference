include("./ht-inf.jl")
include("./mixtures.jl")
using LinearAlgebra
using .HtInf: hitting_times, ht_learn



n = 3
k = 1

"""
g1 = Graphs.uniform_tree(n)
g2 = Graphs.uniform_tree(n)
g3 = Graphs.uniform_tree(n)
m = Mixtures.random([g1, g2]) # , g3])
"""

m = Mixtures.random(n, k)
m.start[:] .= 1 / (n*k)
println("true mixture:")
display(m)

(trails, chains) = Mixtures.sample_trails(m, n_trails=10000, trail_len=10)
Linv = pinv(I - transpose(m.trans[:, :, 1]))
H = hitting_times(Linv)
display(H)

Ms = map(1:k) do i
    M = m.trans[:, :, 1]
    # H = hitting_times(n, trails, chains .== i)
    display(2 .+ M^2 * H)

    println("rest...")
    H_ = hitting_times(n, trails, chains .== i, l=3)
    display(H)

    M = ht_learn(H)
    Mixtures.Mixture(m.start[:, i], M)
end
m_ = Mixtures.concat(Ms...)
println("mixture learned from samples (chain known):")
display(m_)

"""
println("EM:")
m_learn = em(n, k, trails; m_true=m)
display(m_learn)
display(m)
"""


"""
m = Mixtures.random(n, 1)
# m = Mixtures.random([Graphs.uniform_tree(n)])
m.start[:] .= 1 / n
Linv = pinv(I - transpose(m.trans[:, :, 1]))
trails = Mixtures.sample_trails(m, n_trails=10000, trail_len=10)

display(m.trans[:, :, 1])

H = hitting_times(n, trails)
H_ = hitting_times(Linv)
println("true H:")
display(H_)
println("sampled H:")
display(H)

M = ht_learn(H)
println("true M:")
display(m.trans[:, :, 1])
println("computed M:")
display(M)
"""


"""
println("M=")
display(m.trans[:, :, 1])

trails = Mixtures.sample_trails(m, n_trails=1000, trail_len=1000)
trails2 = Mixtures.sample_trails(m, n_trails=500000, trail_len=2)
H = hitting_times(n, trails2)
println("\nsampled:")
display(H)

H2 = hitting_times_stitch(n, trails2)
println("\nsampled (stitch):")
display(H2)

L = I - transpose(m.trans[:, :, 1])
Linv = pinv(L)
println("\ncomputed:")
display(hitting_times(Linv))
"""