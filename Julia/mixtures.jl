module Mixtures

import Graphs
import Graphs.SimpleGraphs
import Base.show
import Hungarian
import StatsBase
import Random
using LinearAlgebra



struct Mixture
    start
    trans
end


function normalize!(m)
    m.start ./= sum(m.start)
    m.trans ./= sum(m.trans, dims=2)
end

function get_rates(m)
    _, k = size(m.start)
    rates = copy(m.trans)
    for i in 1:k
        rates[:, :, i] -= I
    end
    rates
end

function show(io::IO, m::Mixture)
    n, k = size(m.start)
    println("Mixture with ", n, " states and ", k, " chains")
    display(round.(m.start, digits=3))
    display(round.(m.trans, digits=3))
end

function singleton(start, trans)
    start = reshape(start, (size(start)..., 1))
    trans = reshape(trans, (size(trans)..., 1))
    Mixture(start, trans)
end

function concat(ms...)
    start = cat(map(m -> m.start, ms)..., dims=2)
    trans = cat(map(m -> m.trans, ms)..., dims=3)
    Mixture(start, trans)
end

function random(n::Int, k::Int)
    start = rand(n, k)
    start ./= sum(start)
    trans = rand(n, n, k)
    trans ./= sum(trans, dims=2)
    Mixture(start, trans)
end

function random(gs::Vector)
    ms = map(gs) do g
        trans = float.(Matrix(Graphs.adjacency_matrix(g)))
        trans[diagind(trans)] .+= 1
        trans ./= sum(trans, dims=2)
        n, _ = size(trans)
        start = rand(n)
        start ./= sum(start)
        Mixture(start, trans)
    end
    concat(ms...)
end

function distance(m1::Mixture, m2::Mixture)
    @assert size(m1.start) == size(m2.start)
    n, k = size(m1.start)
    D = reshape(m1.trans, (n, n, k, 1)) .- reshape(m2.trans, (n, n, 1, k))
    W = dropdims(sum(abs.(D), dims=(1, 2)), dims=(1, 2))
    _, dist = Hungarian.hungarian(W)
    dist / (2 * n * k)
end

function sample_trails(m::Mixture; n_trails=1::Int, trail_len=10::Int)
    n, k = size(m.start)
    trails = []
    chains = []
    start = reduce(vcat, m.start)
    for _ in 1:n_trails
        x, l = divrem(StatsBase.wsample(start)-1, k)
        M = m.trans[:, :, l+1]
        x += 1
        trail = Array{Int, 1}()
        push!(trails, trail)
        push!(chains, l+1)
        for _ in 1:trail_len
            push!(trail, x)
            x = StatsBase.wsample(M[x, :])
        end
    end
    (trails, chains)
end

function sample_trails_ct(m::Mixture; n_trails=1::Int, trail_len=10::Int)
    n, k = size(m.start)
    rates = get_rates(m)
    trans_ = copy(rates)
    for i in 1:k
        for j in 1:n
            trans_[j, j, i] = 0
        end
    end
    trans = trans_ ./ sum(trans_, dims=2)
    trails = []
    chains = []
    start = reduce(vcat, m.start)
    for _ in 1:n_trails
        x, l = divrem(StatsBase.wsample(start)-1, k)
        K = rates[:, :, l+1]
        K_ = trans_[:, :, l+1]
        M = trans[:, :, l+1]
        x += 1
        t = 0.0
        trail = Array{Tuple{Int, Float64}, 1}()
        push!(trails, trail)
        push!(chains, l+1)
        for _ in 1:trail_len
            push!(trail, (x, t))
            ts = Random.randexp(n) ./ K_[x, :]
            y = argmin(ts)
            t = ts[y]
            """
            y = StatsBase.wsample(M[x, :])
            t = Random.randexp() / K[x, y]
            """
            x = y
        end
    end
    (trails, chains)
end

end
