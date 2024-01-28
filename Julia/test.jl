using Combinatorics
using LinearAlgebra
using Printf
using Graphs
using Graphs.SimpleGraphs
using IterativeSolvers
using LsqFit
using Distributions
using StatsBase
using Laplacians


function rnd_stoch(n)
    M = rand(n, n)
    M ./ sum(M, dims=2)
end

function rnd_graph(n)
    n1 = div(n, 2)
    n2 = n - n1
    sbm = StochasticBlockModel([n1, n2], [0.0 1.0; 1.0 0.0])
    g = SimpleGraph(n, (n1-1)*n2, sbm)
    # g = uniform_tree(n)

    display(adjacency_matrix(g))
    M = Matrix(adjacency_matrix(g))
    M += Diagonal(ones(n))
    M ./ sum(M, dims=2)
end

function stationary(Linv)
    L = pinv(Linv)
    stationary(Linv, L)
end

function stationary(Linv, L)
    p = Linv * L * ones(n)
    d = 1 .- p
    d ./ sum(d)
end

function hitting_times(Linv)
    L = pinv(Linv)
    hitting_times(Linv, L)
end

function hitting_times(Linv, L)
    s = stationary(Linv, L)
    H = Array{Float64}(undef, n, n)
    colsum = sum(Linv, dims=1)
    for u in 1:n
        for v in 1:n
            H[u, v] = colsum[u] - colsum[v] - (Linv[v,u] - Linv[v,v]) / s[v]
        end
    end
    H
end

function hitting_times_fast(Linv, L)
    s = stationary(Linv, L)
    colsum = sum(Linv, dims=1)
    A = transpose(colsum) .- colsum
    B = transpose(Linv .- diag(Linv)) ./ transpose(s)
    H = A - B
    H
end

function L_rk1(Linv, eps, i, j)
    println(">>>")
    base_i = I[i, 1:n]
    base_j = I[j, 1:n]
    c = base_i
    d = base_j - base_i

    A = Linv
    Ainv = pinv(A)
    prod = A * Ainv

    X1 = pinv(A + eps * c * transpose(d))

    beta = 1 + eps * (Ainv[j, i] - Ainv[i, i])
    h = Ainv[j, :] - Ainv[i, :]
    u = - prod[:, i]
    u[i] += 1
    k = Ainv[:, i]
    p2 = - eps * (eps * norm(u)^2 / beta * Ainv * h + k)
    q2 = - norm(h)^2 / beta * eps * transpose(u) - transpose(h)
    sigma2 = eps^2 * norm(h)^2 * norm(u)^2 + beta^2

    X2 = Ainv + eps * Ainv * h * transpose(u) ./ beta - beta/sigma2 * p2 * q2
    x2 = sum(Ainv, dims=2) + eps * sum(u) / beta * Ainv * h - beta/sigma2 * sum(q2) * p2

    # display(X1)
    # display(X2)
    println(">>> DIFF: ", norm(X1 - X2))

    s0 = stationary(Linv)
    Y = Linv + eps * c * transpose(d)
    s1 = 1 .- Y * X1 * ones(n)
    s1 ./= sum(s1)
    s2 = 1 .- Y * x2
    s2 ./= sum(s2)
    ds1 = (s1 - s0) / eps
    ds2 = (s2 - s0) / eps
    ds3 = grad_s(Linv)[i, j, :]
    println(">>> s-DIFF: ", norm(ds1 - ds2), " ", norm(ds1 - ds3), " ", norm(ds2 - ds3))
end

function L_rk1_sketchy(Linv, eps, i, j)
    println(">>>")
    base_i = I[i, 1:n]
    base_j = I[j, 1:n]
    c = base_i
    d = base_j - base_i

    A = Linv
    Ainv = pinv(A)
    prod = A * Ainv

    X1 = pinv(A + eps * c * transpose(d))

    h = Ainv[j, :] - Ainv[i, :]
    u = - prod[:, i]
    u[i] += 1
    k = Ainv[:, i]
    p2 = - eps * k
    q2 = - transpose(h)

    X2 = Ainv + eps * Ainv * h * transpose(u) - p2 * q2
    x2 = sum(Ainv, dims=2) + eps * sum(u) * Ainv * h - sum(q2) * p2

    # display(X1)
    # display(X2)
    println(">>> DIFF: ", norm(X1 - X2))

    s0 = stationary(Linv)
    Y = Linv + eps * c * transpose(d)
    s1 = 1 .- Y * X1 * ones(n)
    s1 ./= sum(s1)
    s2 = 1 .- Y * x2
    s2 ./= sum(s2)
    ds1 = (s1 - s0) / eps
    ds2 = (s2 - s0) / eps
    ds3 = grad_s(Linv)[i, j, :]
    println(">>> s-DIFF: ", norm(ds1 - ds2), " ", norm(ds1 - ds3), " ", norm(ds2 - ds3))
end

function grad_s_sketchy(Linv, L)
    s0 = stationary(Linv, L)
    P = Linv * L
    rowsum_L = dropdims(sum(L, dims=2), dims=2)
    colsum_P = dropdims(sum(P, dims=1), dims=1)
    Lsq = L * transpose(L)
    p0 = Linv * rowsum_L

    ds = zeros(n, n, n)

    """
    eps = 1e-10
    for i = 1:n
        for j = 1:n
            x = rowsum_L + eps * ((1 - colsum_P[i]) * (Lsq[j, :] - Lsq[i, :]) - (rowsum_L[j] - rowsum_L[i]) * L[:, i])
            Linv[i, j] += eps
            Linv[i, i] -= eps
            p1 = Linv * x
            Linv[i, j] -= eps
            Linv[i, i] += eps

            dp = (p1 - p0) / eps
            a = 1 .- p0
            norm_a = norm(a, 1)
            J = a * transpose(ones(n)) / norm_a^2 - I / norm_a
            ds_ij = J * dp

            s1 = 1 .- p1
            s1 ./= sum(s1)
            ds[i, j, :] = (s1 - s0) / eps
            ds[i, j, :] = ds_ij

            # println(">> diff: ", norm((s1 - s0) / eps - ds_ij))
        end
    end
    """

    a = 1 .- p0
    norm_a = norm(a, 1)
    J = repeat(a, 1, n) / norm_a^2 - I / norm_a

    for i = 1:n
        for j = 1:n
            dx = (1 - colsum_P[i]) * (Lsq[j, :] - Lsq[i, :]) - (rowsum_L[j] - rowsum_L[i]) * L[:, i]
            x = rowsum_L

            dp = Linv * dx
            dp[i] += x[j]
            dp[i] -= x[i]
            ds[i, j, :] = J * dp
        end
    end

    return ds
end

function loss(Linv, L, H_train)
    H = hitting_times(Linv, L)
    norm(H - H_train)^2 / 2
end

function grad_loss_naive(Linv, H_train)
    H = hitting_times(Linv)
    s = stationary(Linv)
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
    dloss
end

function grad_loss_fast(Linv, L, H_train)
    H = hitting_times_fast(Linv, L)
    P = Linv * L
    p = P * ones(n)
    d = 1 .- p
    s = d ./ sum(d)
    rowsum_L = dropdims(sum(L, dims=2), dims=2)
    colsum_P = dropdims(sum(P, dims=1), dims=1)
    Lsq = L * transpose(L)
    p0 = Linv * rowsum_L
    H_diff = H - H_train
    rowsum_H_diff = dropdims(sum(H_diff, dims=2), dims=2)
    f = 1 .- 1 ./ s
    yi1 = f .* dropdims(sum(H_diff, dims=1), dims=1)
    yi2 = rowsum_H_diff - diag(H_diff) ./ s
    yi = yi1 - yi2
    yj = rowsum_H_diff - dropdims(sum(H_diff, dims=1), dims=1)
    x = dropdims(sum((Linv .- diag(Linv)) .* transpose(H_diff), dims=2), dims=2) ./ s.^2
    a = 1 .- p0
    norm_a = norm(a, 1)
    Jx = transpose(repeat(a, 1, n) / norm_a^2 - I / norm_a) * x
    LinvJx = transpose(transpose(Linv) * Jx)
    LinvJxL = LinvJx * L
    LinvJxLsq = LinvJx * Lsq

    dx1 = (1 .- colsum_P) .* (LinvJxLsq .- transpose(LinvJxLsq))
    dx2 = (transpose(rowsum_L) .- rowsum_L) .* transpose(LinvJxL)
    dpp = (transpose(rowsum_L) .- rowsum_L) .* Jx
    ds = dx1 - dx2 + dpp

    dloss_part = transpose(yj) .+ yi - transpose(H_diff) ./ s
    dloss = dloss_part + ds

    dloss[diagind(dloss)] .= 0
    dloss
end

"""
function grad_loss_faster(Linv, L, H_train)
    P = Linv * L
    p = sum(P, dims=2)
    d = 1 .- p
    s = d ./ sum(d)

    colsum = sum(Linv, dims=1)
    A = transpose(colsum) .- colsum
    LinvD = Linv .- diag(Linv)
    B = transpose(LinvD) ./ transpose(s)
    H = A - B

    rowsum_L = dropdims(sum(L, dims=2), dims=2)
    colsum_P = dropdims(sum(P, dims=1), dims=1)
    H_diff = H - H_train
    rowsum_H_diff = dropdims(sum(H_diff, dims=2), dims=2)
    colsum_H_diff = dropdims(sum(H_diff, dims=1), dims=1)
    f = 1 .- 1 ./ s
    yi1 = f .* colsum_H_diff
    yi2 = rowsum_H_diff - diag(H_diff) ./ s
    yi = yi1 - yi2
    yj = rowsum_H_diff - colsum_H_diff
    x = dropdims(sum(LinvD .* transpose(H_diff), dims=2), dims=2) ./ s.^2
    p0 = Linv * rowsum_L
    a = 1 .- p0
    norm_a = norm(a, 1)
    Jx = transpose(repeat(a, 1, n) / norm_a^2 - I / norm_a) * x
    LinvJx = transpose(transpose(Linv) * Jx)
    ds = zeros(n, n)
    LinvJxL = LinvJx * L
    LinvJxLsq = LinvJxL * transpose(L)

    dx1 = (1 .- colsum_P) .* (LinvJxLsq .- transpose(LinvJxLsq))
    dx2 = (transpose(rowsum_L) .- rowsum_L) .* transpose(LinvJxL)
    dpp = (transpose(rowsum_L) .- rowsum_L) .* Jx
    ds = dx1 - dx2 + dpp

    dloss_part = transpose(yj) .+ yi - transpose(H_diff) ./ s
    dloss = dloss_part + ds

    dloss[diagind(dloss)] .= 0
    norm(H_diff)^2 / 2, dloss
end
"""

function lstsq(A, b, Ainv)
    # svd(A) \ b
    return Ainv * b

    """
    n, _ = size(A)
    g = Laplacians.complete_graph(n)
    for i = 1:n
        for j = 1:n
            if i == j continue end
            g[i, j] = max(-A[i, j], 0)
        end
    end
    display(g)
    sol = approxchol_lap(g)
    sol(b)
    """
end

function grad_loss_faster(Linv, L, H_train)
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

global total_grad_s_time = 0
function grad_loss(Linv, L, H_train)
    global total_grad_s_time
    H = hitting_times(Linv, L)
    s = stationary(Linv, L)
    total_grad_s_time += @elapsed begin
        # ds = grad_s_sketchy(Linv, L)
        ds = grad_s(Linv)
    end
    H_diff = H - H_train
    rowsum_H_diff = dropdims(sum(H_diff, dims=2), dims=2)

    dloss = zeros(n, n)

    """
    x = zeros(n)
    for v = 1:n
        for u = 1:n
            x[v] += (Linv[v, u] - Linv[v, v]) * (H[u, v] - H_train[u, v]) / s[v]^2
        end
    end
    """
    x = dropdims(sum((Linv .- diag(Linv)) .* transpose(H_diff), dims=2), dims=2) ./ s.^2

    """
    yi = zeros(n)
    for i = 1:n
        f = 1 - 1 / s[i]
        for w = 1:n
            wf = w == i ? f : 1
            yi[i] += (H[w, i] - H_train[w, i]) * f
            yi[i] -= (H[i, w] - H_train[i, w]) * wf
        end
    end
    """
    f = 1 .- 1 ./ s
    yi1 = f .* dropdims(sum(H_diff, dims=1), dims=1)
    yi2 = rowsum_H_diff - diag(H_diff) ./ s
    yi = yi1 - yi2

    """
    yj = zeros(n)
    for j = 1:n
        for w = 1:n
            yj[j] += H[j, w] - H_train[j, w]
            yj[j] -= H[w, j] - H_train[w, j]
        end
    end
    """
    yj = rowsum_H_diff - dropdims(sum(H_diff, dims=1), dims=1)

    for i = 1:n
        for j = 1:n
            if i == j continue end
            dl = 0

            """
            for w = 1:n
                f = 1 - 1 / s[i]
                wf = w == i ? f : 1
                dl += (H[j, w] - H_train[j, w]) * wf
                dl -= H[w, j] - H_train[w, j]
                dl += (H[w, i] - H_train[w, i]) * f
                dl -= (H[i, w] - H_train[i, w]) * wf
            end
            """

            dl += yj[j] + yi[i] - H_diff[j, i] / s[i]

            for v = 1:n
                dl += ds[i, j, v] * x[v]
            end

            dloss[i, j] = dl
        end
    end

    dloss
end

function grad_s(Linv)
    numeric_grad(stationary, Linv)
end

function numeric_grad(loss, Linv, eps=1e-10)
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

function numeric_grad_loss(Linv, H_train, eps=1e-10)
    loss0 = loss(Linv, pinv(Linv), H_train)
    dLinv = Array{Float64}(undef, size(Linv))
    d = zeros(n, n)
    for i = 1:n
        for j = 1:n
            d[i, j] += eps
            d[i, i] -= eps
            dLinv[i, j] = loss(Linv + d, pinv(Linv + d), H_train) - loss0
            d[i, j] -= eps
            d[i, i] += eps
        end
    end
    dLinv ./ eps
end

global schulz_success_early = 0
global schulz_success = 0
global schulz_abort = 0
function schulz(A, Z)
    @assert okay(A)

    global schulz_success
    global schulz_abort
    global schulz_success_early

    for i = 1:20
        Z_new = (2 * I - Z * A) * Z
        if any(isnan.(Z_new))
            schulz_abort += 1
            break
        end
        if all(abs.(Z_new - Z) .< 1e-15)
            schulz_success_early += 1
            return Z_new
        end
        Z = Z_new
    end

    schulz_success += 1
    return pinv(A)
end

function pinvLinv(Linv, prevL, eps=1e-10)
    if prevL === nothing return pinv(Linv) end

    # L = schulz(Linv, prevL)
    L = pinv(Linv)

    """
    @time begin
    L = pinv(Linv)
    end

    @time begin
    # S = LinearAlgebra.svd(Linv)
    S, _ = svdl(Linv, nsv=n-1, vecs=:both)
    s = map(x -> x > eps ? 1/x : 0, S.S)

    L = transpose(S.Vt) * Diagonal(s) * transpose(S.U)
    end
    """

    """
    A = transpose(Linv)
    B = I - ones(n, n) / n
    X = zeros(n, n)
    for i = 1:n
        X[i, :] = gmres(A, B[:, i])
    end
    display(X)
    X
    """
    
    L
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

function okay(x)
    !(any(isnan.(x)) || any(.!isfinite.(x)))
end

global total_loss_grad_time = 0
global total_pinv_time = 0
global total_adam_time = 0

function adam(Linv, loss_grad; eta=1e-3, eps=1e-10, beta1=0.9, beta2=0.999, max_iter=10000, loss_threshold=1e-20, verbose=true)
    global total_loss_grad_time
    global total_pinv_time
    global total_adam_time

    loss_threshold_count = 0
    m = zeros(size(Linv))
    v = zeros(size(Linv))
    L = pinv(Linv)

    for i=1:max_iter
        total_pinv_time += @elapsed begin
            @assert okay(Linv)
            if i % 1 == 0
                L = pinv(Linv)
            end
            # L = pinvLinv(Linv, L)
        end
        total_loss_grad_time += @elapsed begin
            l, g = loss_grad(Linv, L)
            if !okay(g)
                println("recompute pinv!")
                L = pinv(Linv)
                l, g = loss_grad(Linv, L)
            end
            # g .= clamp.(g, -1e10, 1e10)
        end

        total_adam_time += @elapsed begin
            @assert okay(g)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g.^2
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)
            dLinv = eta * m_hat ./ (sqrt.(v_hat) .+ eps)
            Linv -= dLinv
            Linv[diagind(Linv)] -= sum(Linv, dims=2)
        end

        if l < loss_threshold
            loss_threshold_count += 1
        else
            loss_threshold_count = 0
        end
        if loss_threshold_count >= 10 break end

        if verbose && (i % 1000 == 0 || i == 1)
            M = I - transpose(L)
            d  = norm(M - M_true)
            @printf "Iteration %10i: loss=%.10f (%.10f per entry), diff=%.10f (%.10f per entry)\n" i l (l / n^2) d (d / n^2)
        end
    end

    Linv
end

function hitting_times_sample(M, k)
    H = [[[] for _ = 1:n] for _ = 1:n]
    for _ = 1:100000
        h = zeros(n)
        u = sample(1:n)
        v = u
        for i = 0:100
            if h[v] == 0
                push!(H[u][v], i^k)
                h[v] = 1
            end
            v = wsample(M[v, :])
        end
    end
    Array([[sum(H[u][v]) / length(H[u][v]) for v = 1:n] for u = 1:n])
end

function hitting_times_2m(M)
    W = transpose(M)
    L = I - transpose(W)
    Linv = pinv(L)
    H1 = hitting_times(transpose(Linv), transpose(L))
    s = stationary(transpose(Linv), transpose(L))
    # H1_ = hitting_times_sample(M, 2)

    # display(H1_)
    # display(H1)

    v = 1
    A = L
    A[v, :] .= 0
    A[v, v] = 1

    println("first moment:")
    b1 = ones(n)
    b1[v] = 0
    h1 = A \ b1
    d = 1 .- I[1:n,v] / s[v]
    h1_ = [transpose(I[1:n,u] - I[1:n, v]) * Linv * d for u = 1:n]
    display(h1)
    display(h1_)

    println("second moment:")
    b2 = ones(n) + 2 * transpose(W) * h1
    b2[v] = 0
    h2 = A \ b2
    # d = 1 .+ 2 * M * h1 - (1 + 2 * transpose(h1) * W * s) * I[1:n,v] / s[v]
    d = 1 .+ 2 * M * h1 - (1 + 2 * transpose(h1) * s) * I[1:n,v] / s[v]
    h2_ = [transpose(I[1:n,u] - I[1:n, v]) * Linv * d for u = 1:n]
    display(h2)
    display(h2_)

    println("------------")
    # display(M * h1)
    # display(h1 .- 1)
    display(transpose(h1) * s)
    display(sum([transpose(I[1:n,u] - I[1:n, v]) * Linv * (1 .- I[1:n,v] / s[v]) * s[u] for u = 1:n]))
    display(sum([transpose(I[1:n,u] - I[1:n, v]) * Linv * ones(n) * s[u] - transpose(I[1:n,u] - I[1:n, v]) * Linv * I[1:n,v] / s[v] * s[u]  for u = 1:n]))
    display(Linv * ones(n))
end


n = 4

# M_true = [0.03600424950260535 0.1612773856745104 0.16231244819018864 0.050345550174747115 0.05758500734391162 0.09911145278742983 0.12683576433014782 0.09013208612522552 0.05140799419246925 0.1649880616787643; 0.1279216244162943 0.30374074251918487 0.24641509125830918 0.017981889853653826 0.050503965917088556 0.00698280878261917 0.05609587207776268 0.10994704801022592 0.04315826894021233 0.0372526882246492; 0.17954539132196345 0.010594753397221924 0.009200477149307532 0.07803120380306171 0.2541835583749725 0.16858974491696338 0.02777421049981009 0.0535726843191172 0.14947451471188689 0.06903346150569531; 0.18043134765857827 0.060512024411273374 0.22631856682781998 0.03197308350154014 0.057190552631094634 0.13852308963277166 0.004719405721804988 0.05106901957674922 0.011452730894726416 0.23781017914364128; 0.024555674686795825 0.13790664569969252 0.009289718153472595 0.10683466374453904 0.0472455526933794 0.16096927319292975 0.11425472149765728 0.20329712250965679 0.08530838973803631 0.11033823808384052; 0.20218157946813958 0.18703695233210216 0.03000923848302702 0.044412481973746594 0.10692992946151846 0.17776401361900993 0.1073045329696188 0.089196584748383 0.0024006167321941065 0.052764070212260455; 0.1471439063849349 0.11024969112617113 0.12791151587898578 0.11362434717776387 0.0012929384368293242 0.07866617296704413 0.10787822593465805 0.011770253052851778 0.09615225653015189 0.2053106925106092; 0.019905134918919667 0.1861511404395247 0.013333447968214765 0.15978595818391556 0.06625459465475148 0.16455885119389568 0.02176081925975192 0.12737454393376973 0.1421008321872719 0.09877467725998461; 0.02215959414354327 0.018475970283131558 0.13147776653585716 0.07249153421501284 0.0821157805740173 0.23935431938010923 0.1865136923161999 0.07153448038034316 0.048487457416698 0.12738940475508762; 0.16314895930723267 0.058759978664890924 0.13834792165857024 0.03688570368949333 0.15259190512701448 0.01403034116454585 0.12078153706050683 0.13448495255041468 0.05162176708949055 0.1293469336878405]
# M = [0.13600424950260535 0.0612773856745104 0.16231244819018864 0.050345550174747115 0.05758500734391162 0.09911145278742983 0.12683576433014782 0.09013208612522552 0.05140799419246925 0.1649880616787643; 0.1279216244162943 0.30374074251918487 0.24641509125830918 0.017981889853653826 0.050503965917088556 0.00698280878261917 0.05609587207776268 0.10994704801022592 0.04315826894021233 0.0372526882246492; 0.17954539132196345 0.010594753397221924 0.009200477149307532 0.07803120380306171 0.2541835583749725 0.16858974491696338 0.02777421049981009 0.0535726843191172 0.14947451471188689 0.06903346150569531; 0.18043134765857827 0.060512024411273374 0.22631856682781998 0.03197308350154014 0.057190552631094634 0.13852308963277166 0.004719405721804988 0.05106901957674922 0.011452730894726416 0.23781017914364128; 0.024555674686795825 0.13790664569969252 0.009289718153472595 0.10683466374453904 0.0472455526933794 0.16096927319292975 0.11425472149765728 0.20329712250965679 0.08530838973803631 0.11033823808384052; 0.20218157946813958 0.18703695233210216 0.03000923848302702 0.044412481973746594 0.10692992946151846 0.17776401361900993 0.1073045329696188 0.089196584748383 0.0024006167321941065 0.052764070212260455; 0.1471439063849349 0.11024969112617113 0.12791151587898578 0.11362434717776387 0.0012929384368293242 0.07866617296704413 0.10787822593465805 0.011770253052851778 0.09615225653015189 0.2053106925106092; 0.019905134918919667 0.1861511404395247 0.013333447968214765 0.15978595818391556 0.06625459465475148 0.16455885119389568 0.02176081925975192 0.12737454393376973 0.1421008321872719 0.09877467725998461; 0.02215959414354327 0.018475970283131558 0.13147776653585716 0.07249153421501284 0.0821157805740173 0.23935431938010923 0.1865136923161999 0.07153448038034316 0.048487457416698 0.12738940475508762; 0.16314895930723267 0.058759978664890924 0.13834792165857024 0.03688570368949333 0.15259190512701448 0.01403034116454585 0.12078153706050683 0.13448495255041468 0.05162176708949055 0.1293469336878405]
M_true = rnd_stoch(4) # rnd_graph(n)
M = rnd_stoch(n)

L_true = I - transpose(M_true)
Linv_true = pinv(L_true)
L = I - transpose(M)
Linv = pinv(L)

H_true = hitting_times(Linv_true)

if false
    L_ = pinvLinv(Linv, nothing)
    println("DIFFERENCE> ", norm(L_ - L))
end

grad_loss_faster(Linv, L, H_true)
if false
    _, dH1 = grad_loss_faster(Linv, L, H_true)
    dH2 = numeric_grad_loss(Linv, H_true)
    dH3 = grad_loss(Linv, L, H_true)
    # display(dH1)
    # display(dH2)

    println("DIFFERENCE> ", norm(dH1 - dH2))
    println("DIFFERENCE> ", norm(dH1 - dH3))
    println("DIFFERENCE> ", norm(dH2 - dH3))
end

"""
i = 1
j = 3
base_i = I[i, 1:n]
base_j = I[j, 1:n]
L_rk1_sketchy(Linv, 1e-11, i, j)
"""

"""
ds1 = grad_s_sketchy(Linv, L)
ds2 = grad_s(Linv)

println("DIFFERENCE s_grad> ", norm(ds1 - ds2))
"""


if true
    @time begin
        Linv = adam(
            initial_guess(H_true),
            # Linv,
            (Linv, L) -> grad_loss_faster(Linv, L, H_true),
            eta=0.001,
            beta1=0.9,
            beta2=0.99,
            max_iter=20000)
    end

    display(I - transpose(pinv(Linv)))
    display(I - transpose(L_true))

    println("loss_grad time:", total_loss_grad_time)
    println("pinv time:", total_pinv_time)
    println("adam time:", total_adam_time)
    println("schulz: success=", schulz_success, " (early=", schulz_success_early, "), abort=", schulz_abort)
end


# hitting_times_2m(M)


nothing
