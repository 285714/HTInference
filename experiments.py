import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt

from utils import *
import dtlearn as dt
import ctlearn as ct


plt.rcParams.update({
    "text.usetex": True,
    'font.size': 13,
})



def create_graph(n, graph_type, random=False, selfloops=True):
    graph = None
    if graph_type == "complete":
        graph = nx.complete_graph(n)
    elif graph_type == "lollipop":
        m1, r = divmod(n, 2)
        m2 = m1 + r
        graph = nx.lollipop_graph(m1, m2)
    elif graph_type == "grid":
        x = int(np.ceil(np.sqrt(n)))
        graph = nx.grid_graph(dim=(x, x))
        n_rem = x**2 - n
        nodes = list(graph.nodes)
        for i in range(n_rem): graph.remove_node(nodes[i])
    elif graph_type == "star":
        graph = nx.star_graph(n - 1)
    assert(graph is not None)

    mixture = dt.Mixture.random(n, 1)
    mixture.S[:] = 1
    A = nx.to_numpy_array(graph)
    if selfloops: A[range(n), range(n)] = 1
    if random: mixture.Ms[0] *= A
    else: mixture.Ms[0] = A
    mixture.normalize()
    return mixture


@memoize
def test_single_chain(n, graph_type, random, from_trails, n_trails, trail_len, max_iter=10000, seed=None):
    import htinf
    mixture = create_graph(n, graph_type, random)
    H_true = htinf.hitting_times(mixture)

    if from_trails:
        trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
        H_sample = htinf.hitting_times_from_trails(n, trails)
        frob = np.linalg.norm(H_true - H_sample)
    else:
        H_sample = H_true
        frob = 0

    learned_mixture = htinf.ht_learn(H_sample, max_iter=max_iter)
    tv = mixture.recovery_error(learned_mixture)
    return {"tv": tv, "frob": frob}

@genpath
def plot_test_single_chain(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_single_chain(row.n, row.graph_type, row.random, row.from_trails, row.n_trails, row.trail_len, max_iter=row.max_iter, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("graph_type")
    for graph_type, df in grp:
        grp = df.groupby("n")
        x = grp["tv"]
        mean = x.mean()
        std = x.std()

        config = next_config()
        plt.plot(mean.index, mean, label=graph_type, **config)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.legend(loc="upper left")
    if setup['from_trails'][0]:
        plt.title(f"{setup['n_trails'][0]} trails of length {setup['trail_len'][0]}, {setup['max_iter'][0]} iterations")
    else:
        plt.title(f"estimation from true hitting times, {setup['max_iter'][0]} iterations")
    plt.xlabel("$n$")
    plt.ylabel("TV-distance")
    plt.xticks(mean.index)
    savefig()


@memoize
def test_ht_sampling(n, graph_type, random, n_trails, trail_len, seed=None):
    import htinf
    mixture = create_graph(n, graph_type, random)
    H_true = htinf.hitting_times(mixture)
    trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
    H_sample = htinf.hitting_times_from_trails(n, trails)
    frob = np.linalg.norm(H_true - H_sample)
    return {"frob": frob}


@genpath
def plot_test_ht_sampling(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_ht_sampling(row.n, row.graph_type, row.random, row.n_trails, row.trail_len, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby(["n_trails", "trail_len"])
    x = grp["frob"]
    mean = x.mean()
    df = pd.DataFrame(mean).reset_index()
    l = len(setup["n_trails"])
    X = df.n_trails.to_numpy().reshape(l, -1)
    Y = df.trail_len.to_numpy().reshape(l, -1)
    Z = df.frob.to_numpy().reshape(l, -1)

    # plt.figure(figsize=(8, 3.8))
    from matplotlib import cbook, cm
    from matplotlib.colors import LightSource
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ax.set_xlabel('number of trails')
    ax.set_ylabel('trail length')
    ax.set_zlabel('error')

    ax.plot_surface(X, Y, Z, facecolors=rgb)

    plt.title(setup['graph_type'][0])
    savefig()


@memoize
def test_mixture(n, k, n_trails, trail_len, num_iters=100, seed=None):
    import htinf
    mixture = dt.Mixture.random(n, k)
    mixture.S[:] = 1
    mixture.normalize()
    trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
    learned_mixture = htinf.em(n, k, trails, num_iters=num_iters)
    recovery_error = mixture.recovery_error(learned_mixture)
    return {"recovery_error": recovery_error}

@genpath
def plot_test_mixture(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_mixture(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby(["n"])
    x = grp["recovery_error"]
    mean = x.mean()


# @memoize
def nba_ht(k, team, season, num_iters=100, verbose=True):
    import NBA.extract as nba

    df, state_dict = nba.load_trails(team, season, tau=0.1, model_score=True, verbose=verbose)
    trails = [row.trail_ct + [(1 if row.ptsScored > 0 else 0, 1)] for _, row in df.iterrows()]

    import htinf
    n = len(state_dict)
    mixture = htinf.em(n, k, trails, num_iters=num_iters)

    print(mixture)

    """
    trails_dt = np.array(df.trail_dt.tolist())
    n = len(state_dict)

    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=10000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    tau = 0.1
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=1000)

    trails_ct_ = [row.trail_ct + [(1 if row.ptsScored > 0 else 0, 1)] for _, row in df.iterrows()]
    mixture_ct_ = ct.continuous_em_learn(n, L, trails_ct_, max_iter=100)

    trails_ct = df.trail_ct.tolist()
    lls_ct = ct.likelihood(mixture_ct, trails_ct)
    lls_ct_ = ct.likelihood(mixture_ct_, trails_ct)
    ls_ct = np.exp(lls_ct)
    ls_ct_ = np.exp(lls_ct_)
    explainability = np.sum(ls_ct, axis=0)
    explainability_ = np.sum(ls_ct_, axis=0)
    ls_ct_sorted = np.sort(ls_ct, axis=0)[::-1, :]
    ls_ct_sorted_ = np.sort(ls_ct_, axis=0)[::-1, :]
    discrimination = ls_ct_sorted[0, :] - ls_ct_sorted[1, :]
    discrimination_ = ls_ct_sorted_[0, :] - ls_ct_sorted_[1, :]

    prediction_error = nba_prediction_error(df, mixture_ct)
    prediction_error_ = nba_prediction_error(df, mixture_ct_)

    # ct.likehood(em_mixture_ct, trails)

    return {
        "em_explainability": explainability,
        "em_discrimination": discrimination,
        "continuous_em_explainability": explainability_,
        "continuous_em_discrimination": discrimination_,
        "em_prediction_error": prediction_error,
        "continuous_em_prediction_error": prediction_error_,
    }
    """

