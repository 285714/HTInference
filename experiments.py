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
        x = int(np.sqrt(n))
        assert(x**2 == n)
        graph = nx.grid_graph(dim=(x, x))
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
def test_single_chain(n, graph_type, random, from_trails, n_trails, trail_len, seed=None):
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

    learned_mixture = htinf.ht_learn(H_sample)
    tv = mixture.recovery_error(learned_mixture)
    return {"tv": tv, "frob": frob}

@genpath
def plot_test_single_chain(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_single_chain(row.n, row.graph_type, row.random, row.from_trails, row.n_trails, row.trail_len, seed=row.seed),
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
        plt.title(f"{setup['graph_type'][0]}: {setup['n_trails'][0]} trails of length {setup['trail_len'][0]}")
    else:
        plt.title(f"{setup['graph_type'][0]}: estimation from true hitting times")
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

