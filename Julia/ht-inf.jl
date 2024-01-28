module HtInf

include("./mixtures.jl")
import Graphs
using LinearAlgebra
import StatsBase
using Printf



function to_mixture(ss, ts)
    start = cat(ss..., dims=2)
    trans = cat(ts..., dims=3)
    Mixtures.Mixture(start, trans)
end

function from_mixture(m)
    _, k = size(m.start)
    return ([m.start[:, i] for i in 1:k], [m.trans[:, :, i] for i in 1:k])
end

function get_trails(m, n_trails, trail_len)
    # m = Mixtures.singleton(start, trans)
    Mixtures.sample_trails(m, n_trails=n_trails, trail_len=trail_len)
end

function hitting_times_ct(n::Int, trails, weights=ones(length(trails)))
    ht_sum = zeros(n, n)
    count = zeros(n, n)
    h = Array{Bool}(undef, n)
    for (trail, weight) in zip(trails, weights)
        h[:] .= false
        u = trail[1]
        time = 0
        for (i, (v, t)) in Iterators.enumerate(trail)
            time += t
            if !h[v]
                ht_sum[u, v] += weight * time
                count[u, v] += weight
                h[v] = true
            end
        end
    end
    # println("count=")
    #display(count)
    ht_sum ./ count
end

function hitting_times(n::Int, trails, weights=ones(length(trails)))
    ht_sum = zeros(n, n)
    count = zeros(n, n)
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

function adam(Linv, loss_grad; eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=10000, loss_threshold=1e-5, verbose=false)
    global total_loss_grad_time
    global total_pinv_time
    global total_adam_time

    loss_threshold_count = 0
    m = zeros(size(Linv))
    v = zeros(size(Linv))
    L = pinv(Linv)

    for i=1:max_iter
        L = pinv(Linv)
        l, g = loss_grad(Linv, L)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g.^2
        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)
        dLinv = eta * m_hat ./ (sqrt.(v_hat) .+ eps)
        Linv -= dLinv
        Linv[diagind(Linv)] -= sum(Linv, dims=2)

        if l < loss_threshold
            loss_threshold_count += 1
        else
            loss_threshold_count = 0
        end
        if loss_threshold_count >= 10 break end

        if verbose && (i % 1000 == 0 || i == 1)
            M = I - transpose(L)
            d = norm(M - M_true)
            @printf "Iteration %10i: loss=%.10f (%.10f per entry), diff=%.10f (%.10f per entry)\n" i l (l / n^2) d (d / n^2)
        end
    end

    Linv
end

function initial_guess(H)
    X = 1 ./ H
    X[diagind(X)] .= 0
    X[diagind(X)] = 1 .- sum(X, dims=2)
    # X + randn(n, n) * 1e-5
    M = X ./ sum(X, dims=2)
    L = I - transpose(M)
    Linv = pinv(L)
    Linv
end

function ht_learn(H_true; max_iter=10000, verbose=false)
    Linv = adam(
        initial_guess(H_true),
        (Linv, L) -> grad_loss_faster(Linv, L, H_true);
        verbose=verbose,
        max_iter=max_iter)
    L = pinv(Linv)
    M = transpose(I - L)
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


function em(n, k, trails; num_iters=100, m_true=nothing, verbose=false, hitting_times=hitting_times)
    m = Mixtures.random(n, k)
    m.start[:] .= 1 / n
    for iter in 1:num_iters
        ll = rel_likelihood(m, trails)

        Hs = map(1:k) do i
            hitting_times(n, trails, ll[:, i])
        end

        map(1:k) do i
            m.trans[:, :, i] = ht_learn(Hs[i])
        end

        m.trans .= abs.(m.trans)
        m.trans ./= sum(m.trans, dims=2)

        if verbose
            println("\n\nIteration ", iter)
            if !isnothing(m_true)
                println("distance=", Mixtures.distance(m, m_true))
            end
            display(m.trans)
        end
    end
    m
end

function em_ct(n, k, trails; num_iters=100, m_true=nothing, verbose=false)
    em(n, k, trails; num_iters=100, m_true=m_true, verbose=verbose, hitting_times=hitting_times_ct)
end


end
