module HtInf

include("./mixtures.jl")
import Graphs
using LinearAlgebra
import StatsBase
using Printf
import Hungarian



function distance(m1, m2)
    @assert size(m1.start) == size(m2.start)
    n, k = size(m1.start)
    D = reshape(m1.trans, (n, n, k, 1)) .- reshape(m2.trans, (n, n, 1, k))
    W = dropdims(sum(abs.(D), dims=(1, 2)), dims=(1, 2))
    _, dist = Hungarian.hungarian(W)
    dist / (2 * n * k)
end

function to_mixture(ss, ts)
    start = cat(ss..., dims=2)
    trans = cat(ts..., dims=3)
    Mixtures.Mixture(start, trans)
end

function from_mixture(m)
    _, k = size(m.start)
    return ([m.start[:, i] for i in 1:k], [m.trans[:, :, i] for i in 1:k])
end

function get_trails_ct(m, n_trails, trail_len)
    # m = Mixtures.singleton(start, trans)
    Mixtures.sample_trails_ct(m, n_trails=n_trails, trail_len=trail_len)
end

function get_trails(m, n_trails, trail_len)
    # m = Mixtures.singleton(start, trans)
    Mixtures.sample_trails(m, n_trails=n_trails, trail_len=trail_len)
end

function hitting_times_ct_naive(n::Int, trails, weights=ones(length(trails)))
    ht_sum = zeros(n, n)
    count = 0.00001 * ones(n, n)
    h = Array{Bool}(undef, n)
    for (trail, weight) in zip(trails, weights)
        # println(trail)
        h[:] .= false
        u, _ = trail[1]
        time = 0
        for (i, (v, t)) in Iterators.enumerate(trail)
            time += t
            if !h[v]
                ht_sum[u, v] += weight * time
                count[u, v] += weight
                h[v] = true
                # println(u, " -> ", v, " (", time, ")")
            end
        end
    end
    # println("count=")
    # display(count)
    ht_sum ./ count
end

function hitting_times_ct(n::Int, trails, weights=ones(length(trails)))
    ht_sum = zeros(n, n)
    count = 0.00001 * ones(n, n)

    for (trail, weight) in zip(trails, weights)
        times = [Array{Float64}(undef, 0) for _ in 1:n]
        time = 0
        for (i, (v, t)) in Iterators.enumerate(trail)
            time += t
            push!(times[v], time)
        end

        indices = ones(Int, n)
        time = 0
        for (i, (u, t)) in Iterators.enumerate(trail)
            time += t
            for v in 1:n
                if indices[v] <= length(times[v])
                    ht_sum[u, v] += weight * (times[v][indices[v]] - time)
                    count[u, v] += weight
                end
            end
            indices[u] += 1
        end
    end
    # println("count=")
    # display(count)
    ht_sum ./ count
end

function hitting_times(n::Int, trails, weights=ones(length(trails)))
    ht_sum = zeros(n, n)
    count = 0.00001 * ones(n, n)

    for (trail, weight) in zip(trails, weights)
        positions = [Array{Int}(undef, 0) for _ in 1:n]
        for (i, v) in Iterators.enumerate(trail)
            push!(positions[v], i)
        end

        indices = ones(Int, n)
        for (i, u) in Iterators.enumerate(trail)
            for v in 1:n
                if indices[v] <= length(positions[v])
                    ht_sum[u, v] += weight * (positions[v][indices[v]] - i)
                    count[u, v] += weight
                end
            end
            indices[u] += 1
        end
    end
    # println("count=")
    # display(count)
    ht_sum ./ count
end

function hitting_times_naive(n::Int, trails, weights=ones(length(trails)))
    ht_sum = zeros(n, n)
    count = 0.00001 * ones(n, n)
    h = Array{Bool}(undef, n)
    for (trail, weight) in zip(trails, weights)
        h[:] .= false
        u = trail[1]
        for (i, v) in Iterators.enumerate(trail)
            if !h[v]
                ht_sum[u, v] += weight * (i - 1)
                count[u, v] += weight
                h[v] = true
            end
        end
    end
    # println("count=")
    #display(count)
    ht_sum ./ count
end

"""
function hitting_times(n::Int, trails, weights=ones(length(trails)); l=2)
    probs_count = zeros(n, n)
    for (trail, weight) in zip(trails, weights)
        for (x, y) in zip(trail[1:end-l+1], trail[l:end])
            probs_count[x, y] += weight
        end
    end
    probs = probs_count ./ sum(probs_count, dims=2)

    s_count = zeros(n)
    for (trail, weight) in zip(trails, weights)
        s_count[trail[end]] += weight
    end
    s = s_count ./ sum(s_count)

    Linv = pinv(I - transpose(probs))
    s = stationary(Linv)

    colsum = sum(Linv, dims=1)
    A = transpose(colsum) .- colsum
    B = transpose(Linv .- diag(Linv)) ./ transpose(s)
    H = A - B
    (l-1) * H
end
"""

function hitting_times(Linv::Matrix)
    n, _ = size(Linv)
    L = pinv(Linv)
    hitting_times(Linv, L)
end

function hitting_times(Linv::Matrix, L::Matrix)
    s = stationary(Linv, L)
    colsum = sum(Linv, dims=1)
    A = transpose(colsum) .- colsum
    B = transpose(Linv .- diag(Linv)) ./ transpose(s)
    H = A - B
    H
end

function stationary(Linv::Matrix)
    L = pinv(Linv)
    stationary(Linv, L)
end

function stationary(Linv::Matrix, L::Matrix)
    n, _ = size(Linv)
    p = Linv * L * ones(n)
    d = 1 .- p
    d ./ sum(d)
end

function lstsq(A, b, Ainv)
    # svd(A) \ b
    return Ainv * b
end

function grad_loss_faster(Linv, L, H_train)
    n, _ = size(Linv)
    rowsum_L = lstsq(Linv, ones(n), L)
    p = Linv * rowsum_L
    d = 1 .- p
    norm_d = norm(d, 1)
    s = d ./ norm_d

    colsum = sum(Linv, dims=1)
    A = transpose(colsum) .- colsum
    LinvD = Linv .- diag(Linv)
    B = transpose(LinvD) ./ transpose(s)
    H = A - B

    colsum_P = lstsq(transpose(Linv), transpose(colsum), transpose(L))
    H_diff = H - H_train
    H_diff_s = H_diff ./ transpose(s)
    rowsum_H_diff = dropdims(sum(H_diff, dims=2), dims=2)
    colsum_H_diff = dropdims(sum(H_diff, dims=1), dims=1)
    f = 1 .- 1 ./ s
    yi1 = f .* colsum_H_diff
    yi2 = rowsum_H_diff - diag(H_diff_s)
    yi = yi1 - yi2
    yj = rowsum_H_diff - colsum_H_diff
    x = dropdims(sum(H_diff_s .* B, dims=1), dims=1)
    Jx = (transpose(s) * x .- x) / norm_d
    LinvJx = transpose(transpose(Linv) * Jx)
    LinvJxL = lstsq(transpose(Linv), transpose(LinvJx), transpose(L))
    LinvJxLsq = lstsq(Linv, LinvJxL, L)

    dx1 = (1 .- colsum_P) .* (transpose(LinvJxLsq) .- LinvJxLsq)
    ds = dx1 + (transpose(rowsum_L) .- rowsum_L) .* (Jx - LinvJxL)

    dloss_part = transpose(yj) .+ yi - transpose(H_diff_s)
    dloss = dloss_part + ds

    dloss[diagind(dloss)] .= 0
    norm(H_diff)^2 / 2, dloss
end

function adam(Linv, loss_grad; eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=10000, loss_threshold=1e-5, verbose=false, return_loss=false)
    n, _ = size(Linv)

    """
    if verbose
        println("\n[ADAM] Init")
        display(transpose(I - pinv(Linv)))
    end
    """

    loss_threshold_count = 0
    m = zeros(size(Linv))
    v = zeros(size(Linv))
    L = pinv(Linv)
    min_loss_Linv = Linv
    min_loss = Inf

    for i=1:max_iter
        L = pinv(Linv)
        l, g = loss_grad(Linv, L)

        if l < min_loss
            min_loss = l
            min_loss_Linv = Linv
        end

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g.^2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dLinv = eta * m_hat ./ (sqrt.(v_hat) .+ eps)
        Linv -= dLinv

        if l < loss_threshold
            loss_threshold_count += 1
        else
            loss_threshold_count = 0
        end
        if loss_threshold_count >= 10 break end

        if verbose && (i % 10000 == 0 || i == 1)
            M = I - transpose(L)
            println("\n[ADAM] Iteration ", i, ": loss=", l, " num-err=", sum(abs.(sum(Linv, dims=2))))
            # display(M)
            # d = norm(M - M_true)
            # @printf "Iteration %10i: loss=%.10f (%.10f per entry), diff=%.10f (%.10f per entry)\n" i l (l / n^2) d (d / n^2)
        end

        # Linv[diagind(Linv)] -= sum(Linv, dims=2)
        Linv .-= sum(Linv, dims=2) / n
    end

    if return_loss
        return (min_loss_Linv, min_loss)
    else
        min_loss_Linv
    end
end

function initial_guess(H)
    n, _ = size(H)
    X = rand(n, n)

    # X = 1 ./ H

    """
    X = 1 ./ H
    X[diagind(X)] .= 0
    X[diagind(X)] = 1 .- sum(X, dims=2)
    # X + randn(n, n) * 1e-5
    """

    # println(">>>")
    # display(X)
    M = X ./ sum(X, dims=2)
    L = I - transpose(M)
    Linv = pinv(L)
    Linv
end

function ht_learn(H_true; max_iter=10000, verbose=true, return_loss=false, return_time=false)
    time = @elapsed begin
        Linv, loss = adam(
            initial_guess(H_true),
            (Linv, L) -> grad_loss_faster(Linv, L, H_true);
            eta=1e-4,
            beta1=0.99,
            verbose=verbose,
            max_iter=max_iter,
            return_loss=true)
    end
    L = pinv(Linv)
    M = transpose(I - L)

    if return_loss && return_time
        return (M, loss, time)
    elseif return_loss
        return (M, loss)
    elseif return_time
        return (M, time)
    else
        return M
    end
end


function log_likelihood(start::Vector, trans::Matrix, trails)
    map(trails) do trail
        x = trail[1]
        log_ll = log(start[x])
        for y in trail[2:end]
            log_ll += log(trans[x, y])
            x = y
        end
        log_ll
    end
end

function log_likelihood(m::Mixtures.Mixture, trails)
    _, k = size(m.start)
    log_ll = map(1:k) do i
        log_likelihood(m.start[:, i], m.trans[:, :, i], trails)
    end
    hcat(log_ll...)
end

function rel_likelihood(m::Mixtures.Mixture, trails)
    log_ll = log_likelihood(m, trails)
    rel_log_ll = log_ll .- maximum(log_ll, dims=2)
    x = exp.(rel_log_ll)
    rel_ll = x ./ sum(x, dims=2)
end


function log_likelihood_ct(start::Vector, rates::Matrix, trails)
    rates[diagind(rates)] .= 1
    log_rates = log.(rates)
    map(trails) do trail
        (x, _) = trail[1]
        log_ll = log(start[x])
        for (y, t) in trail[2:end]
            log_ll += log_rates[x, y] - rates[x, y] * t
            x = y
        end
        log_ll
    end
end

function log_likelihood_ct(m::Mixtures.Mixture, trails)
    _, k = size(m.start)
    rates = Mixtures.get_rates(m)
    log_ll = map(1:k) do i
        log_likelihood_ct(m.start[:, i], rates[:, :, i], trails)
    end
    hcat(log_ll...)
end

function rel_likelihood_ct(m::Mixtures.Mixture, trails)
    log_ll = log_likelihood_ct(m, trails)
    rel_log_ll = log_ll .- maximum(log_ll, dims=2)
    x = exp.(rel_log_ll)
    rel_ll = x ./ sum(x, dims=2)
end


function learn_start(n, trails, weights)
    s = zeros(n)
    for (trail, weight) in zip(trails, weights)
        u = trail[1]
        s[u] += weight
    end
    s / length(trails)
end

function learn_start_ct(n, trails, weights)
    s = zeros(n)
    for (trail, weight) in zip(trails, weights)
        u, _ = trail[1]
        s[u] += weight
    end
    s / length(trails)
end


function em(n, k, trails; num_iters=100, m_true=nothing, verbose=true,
               rel_likelihood=rel_likelihood, hitting_times=hitting_times,
               learn_start=learn_start)
    if trails isa Matrix
        trails = [trails[i, :] for i in 1:size(trails,1)]
    end
    println("learn_start=", learn_start)

    m = Mixtures.random(n, k)
    m.start[:] .= 1 / (n * k)
    for iter in 1:num_iters
        ll = rel_likelihood(m, trails)

        Hs = map(1:k) do i
            hitting_times(n, trails, ll[:, i])
        end

        losses = []
        map(1:k) do i
            if !(learn_start isa Bool)
                m.start[:, i] = learn_start(n, trails, ll[:, i])
            end
            m.trans[:, :, i], l = ht_learn(Hs[i], return_loss=true, verbose=false)
            push!(losses, l)
        end

        m.trans .= abs.(m.trans)
        m.trans ./= sum(m.trans, dims=2)

        if verbose
            println("\n\n[EM] Iteration ", iter, " losses=", losses)
            if !isnothing(m_true)
                println("distance=", distance(m, m_true))
            end
            # display(m.trans)
        end
    end
    m
end

function em_ct(n, k, trails; num_iters=100, m_true=nothing, verbose=true, learn_start=learn_start_ct)
    em(n, k, trails; num_iters=num_iters, m_true=m_true, verbose=verbose, rel_likelihood=rel_likelihood_ct, hitting_times=hitting_times_ct, learn_start=learn_start)
end



function grad_loss_naive(Linv, L, H_train)
    n, _ = size(Linv)
    H = hitting_times(Linv, L)
    s = stationary(Linv, L)
    ds = grad_s(Linv)

    dH = zeros(n, n, n, n)
    dloss = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if i == j continue end
            for u in 1:n
                for v in 1:n
                    if u == v continue end

                    x = (j == u) + (i == v) - (i == u) - (j == v)
                    a = x - (v == i ? x / s[v] : 0)
                    b = ds[i, j, v] * (Linv[v, u] - Linv[v, v]) / s[v]^2
                    dH[i, j, u, v] = a + b

                    dloss[i, j] += dH[i, j, u, v] * (H[u, v] - H_train[u, v])
                end
            end
        end
    end
    Inf, dloss
end

function grad_s(Linv)
    numeric_grad(stationary, Linv)
end

function numeric_grad(loss, Linv, eps=1e-10)
    n, _ = size(Linv)
    loss0 = loss(Linv)
    dLinv = Array{Float64}(undef, size(Linv)..., size(loss0)...)
    d = zeros(n, n)
    for i = 1:n
        for j = 1:n
            d[i, j] += eps
            d[i, i] -= eps
            dLinv[i, j, CartesianIndices(loss0)] = loss(Linv + d) - loss0
            d[i, j] -= eps
            d[i, i] += eps
        end
    end
    dLinv ./ eps
end

function time_htlearn_naive(H_true; max_iter=10000)
    time = @elapsed begin
        adam(
            initial_guess(H_true),
            (Linv, L) -> grad_loss_naive(Linv, L, H_true);
            verbose=false,
            max_iter=max_iter)
    end
    return time
end



end
