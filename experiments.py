import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment

from utils import *
import dtlearn as dt
import ctlearn as ct


plt.rcParams.update({
    "text.usetex": True,
    'font.size': 14,
})

graph_names = {
    "complete": "$K_{n}$",
    "lollipop": "LOL$_n$",
    "grid": "$G_{\sqrt n \\times \sqrt n}$",
    "star": "$S_{n}$",
}



def create_mixture_dt(n, k, mix_type=None):
    if mix_type == None:
        return dt.Mixture.random(n, k)
    elif mix_type == "stargrid":
        assert(k == 2)
        m1 = create_graph(n, "star")
        m2 = create_graph(n, "grid")
        m = dt.Mixture(np.array([m1.S[0], m2.S[0]]), np.array([m1.Ms[0], m2.Ms[0]]))
        m.normalize()
        return m


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
    if graph_type == "dag":
        graph = nx.DiGraph()
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(i, j)
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
def test_dag(n, max_iter, noise_std=0, noise="homo", seed=None):
    import htinf
    mixture = create_graph(n, "dag", selfloops=False)
    mixture_ = dt.Mixture(np.copy(mixture.S), np.copy(mixture.Ms))
    mixture_.Ms += 0.01
    mixture_.normalize()
    H_true = htinf.hitting_times(mixture_)
    # import pdb; pdb.set_trace()
    if noise == "homo":
        H_noise = H_true + np.random.normal(0, noise_std, size=H_true.shape)
    else:
        H_noise = H_true + np.random.normal(0, H_true * noise_std / noise)
    learned_mixture, loss = htinf.ht_learn(H_noise, max_iter=max_iter, return_loss=True)
    M = learned_mixture.Ms[0]
    M_ = M * (M > (0.1 * np.max(M, axis=1))[:, None])
    learned_mixture_ = dt.Mixture(learned_mixture.S, np.array([M_]))
    learned_mixture_.normalize()
    import pdb; pdb.set_trace()
    tv2 = mixture.recovery_error(learned_mixture)
    tv = mixture.recovery_error(learned_mixture_)
    print(mixture)
    print(learned_mixture)
    return {"tv2": tv2, "tv": tv, "loss": loss, "mixture": learned_mixture, "mixture_": learned_mixture_}

@genpath
def plot_test_dag(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_dag(row.n, row.max_iter, noise_std=row.noise_std, noise=row.noise, seed=row.seed),
                  axis=1, result_type='expand'))

    from networkx.drawing.nx_agraph import write_dot

    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 8,
    })

    import matplotlib.colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "red", "red","red", "red", "red"])

    for i, x in df.iterrows():
        if i == 0: continue
        # G = nx.DiGraph()
        M = x.mixture_.Ms[0]
        n = x.n
        fig, ax = plt.subplots()
        im = ax.imshow(M, cmap=cmap) # "Wistia"
        ax.set_xticks(np.arange(n), labels=range(1, n+1))
        ax.set_yticks(np.arange(n), labels=range(1, n+1))

        for u in range(n):
            for v in range(n):
                text = ax.text(v, u, np.round(M[u, v], 2), ha="center", va="center", color="w")

        fig.tight_layout()
        savefig()

        import pdb; pdb.set_trace()
        return


        """
        n = x.n
        for u in range(n):
            for v in range(n):
                if M[u, v] > 0:
                    G.add_edge(u, v, weight=M[u, v])
        write_dot(G, f"dag{i}.dot")
        """


    """
    import pdb; pdb.set_trace()
    grp = df.groupby("noise_std")
    fig, ax1 = plt.subplots(figsize=(6.1, 3.8))
    ax2 = ax1.twinx()

    x = grp["tv"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    ax1.plot(mean.index, mean, label="recovery_error", **config)
    ax1.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["loss"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    ax2.plot(mean.index, mean, label="loss", **config)
    ax2.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=config["color"])

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_xlabel("noise $\sigma$")
    ax1.set_ylabel("recovery error")
    ax2.set_ylabel("loss")
    plt.xticks(mean.index[1:])
    savefig()

    mixture_min = df[df.noise_std == df.noise_std.min()].iloc[0].mixture
    mixture_max = df[df.noise_std == df.noise_std.max()].iloc[0].mixture
    print(mixture_min)
    print(mixture_max)
    """


@memoize
def test_grid(n, max_iter, noise_std=0, noise="homo", seed=None):
    import htinf
    mixture = create_graph(n, "grid", selfloops=True)
    H_true = htinf.hitting_times(mixture)
    if noise == "homo":
        H_noise = H_true + np.random.normal(0, noise_std, size=H_true.shape)
    else:
        H_noise = H_true + np.random.normal(0, H_true * noise_std / noise)
    learned_mixture, loss = htinf.ht_learn(H_noise, max_iter=max_iter, return_loss=True)
    tv = mixture.recovery_error(learned_mixture)
    print(mixture)
    print(learned_mixture)
    return {"tv": tv, "loss": loss, "mixture": learned_mixture}

@genpath
def plot_test_grid(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_grid(row.n, row.max_iter, noise_std=row.noise_std, noise=row.noise, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("noise_std")
    fig, ax1 = plt.subplots() # figsize=(6.1, 3.8)
    ax2 = ax1.twinx()

    x = grp["tv"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    ax1.plot(mean.index, mean, label="recovery_error", **config)
    ax1.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    x = grp["loss"]
    mean = x.mean()
    std = x.std()
    config = next_config()
    ax2.plot(mean.index, mean, label="loss", **config)
    ax2.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=config["color"])

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_xlabel("noise $\sigma$")
    ax1.set_ylabel("recovery error")
    ax2.set_ylabel("loss")
    plt.xticks(mean.index[1:])
    savefig()

    mixture_min = df[df.noise_std == df.noise_std.min()].iloc[0].mixture
    mixture_max = df[df.noise_std == df.noise_std.max()].iloc[0].mixture
    print(mixture_min)
    print(mixture_max)


@memoize
def test_runtime(n, max_iter, seed=None):
    import htinf
    import htinfprev
    mixture = create_graph(n, "complete", selfloops=True, random=True)
    H_true = htinf.hitting_times(mixture)
    htinf.ht_learn(H_true, max_iter=max_iter, return_time=True) # compile once
    _, anal_time = htinf.ht_learn(H_true, max_iter=max_iter, return_time=True)
    naive_time = 0 # htinf.time_htlearn_naive(H_true, max_iter=max_iter)
    num_time = 0 # htinfprev.time_htlearn_numeric(H_true, max_iter=max_iter)
    return {"anal_time": anal_time, "naive_time": naive_time, "num_time": num_time}

def show_runtimes(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_runtime(row.n, row.max_iter, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("n")
    mean = grp.mean()
    std = grp.std()

    print(mean)
    print(std)


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

    learned_mixture, loss = htinf.ht_learn(H_sample, max_iter=max_iter, return_loss=True)
    tv = mixture.recovery_error(learned_mixture)
    return {"tv": tv, "frob": frob, "loss": loss}

@genpath
def plot_test_single_chain(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_single_chain(row.n, row.graph_type, row.random, row.from_trails, row.n_trails, row.trail_len, max_iter=row.max_iter, seed=row.seed),
                  axis=1, result_type='expand'))

    # plt.figure()
    # print(df)
    fig, ax1 = plt.subplots() # figsize=(6.1, 3.4)
    ax2 = ax1.twinx()

    grp = df.groupby("graph_type")
    for graph_type, df in grp:
        grp = df.groupby("n")

        x = grp["tv"]
        mean = x.mean()
        std = x.std()
        config = next_config()
        ax1.plot(mean.index, mean, label="recovery error", **config)
        ax1.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        ax1.set_ylim(bottom=-0.005, top=0.155)

        x = grp["loss"]
        mean = x.mean()
        std = x.std()
        config = next_config()
        ax2.plot(mean.index, mean, label="loss", **config)
        ax2.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=config["color"])
        ax2.set_ylim(bottom=-13300, top=413300)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    """
    if setup['from_trails'][0]:
        plt.title(f"{setup['n_trails'][0]} trails of length {setup['trail_len'][0]}, {setup['max_iter'][0]} iterations")
    else:
        plt.title(f"estimation from true hitting times, {setup['max_iter'][0]} iterations")
    """
    ax1.set_xlabel("number of states $n$")
    ax1.set_ylabel("recovery error")
    ax2.set_ylabel("loss")
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
    return {"frob": frob, "cover_time": np.max(H_true)}


@genpath
def plot_test_ht_sampling(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_ht_sampling(row.n, row.graph_type, row.random, row.n_trails, row.trail_len, seed=row.seed),
                  axis=1, result_type='expand'))

    # title = f"{setup['graph_type'][0]} on {setup['n'][0]} vertices with cover time {df['cover_time'].iloc[0]:.1f}"
    title = graph_names[setup['graph_type'][0]]

    # plt.figure(figsize=(3.4, 3.4))

    grp = df.groupby(["n_trails", "trail_len"])
    x = grp["frob"]
    mean = x.mean()
    df = pd.DataFrame(mean).reset_index()
    l = len(setup["n_trails"])
    X = df.n_trails.to_numpy().reshape(l, -1)
    Y = df.trail_len.to_numpy().reshape(l, -1)
    Z = df.frob.to_numpy().reshape(l, -1)

    # plt.figure(figsize=(1, 1))
    from matplotlib import cbook, cm
    from matplotlib.colors import LightSource
    ls = LightSource(270, 45)
    rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(3.4, 3.4))

    ax.set_xlabel('number of trails')
    ax.set_ylabel('trail length', linespacing=2)
    ax.set_zlabel('error')

    ax.plot_surface(X, Y, Z, facecolors=rgb)

    plt.gca().invert_yaxis()

    plt.title(title, y=1.0)
    savefig()


@memoize
def test_mixture_dt(n, k, n_trails, trail_len, num_iters=100, mix_type=None, seed=None):
    import htinf
    mixture = create_mixture_dt(n, k, mix_type) # dt.Mixture.random(n, k)
    mixture.S[:] = 1
    mixture.normalize()
    trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
    learned_mixture = htinf.em(n, k, trails, num_iters=num_iters)
    recovery_error = mixture.recovery_error(learned_mixture)
    return {"recovery_error": recovery_error,
            **test_mixture_dt_baseline(n, k, n_trails, trail_len, num_iters=num_iters, seed=seed)}


@memoize
def test_mixture_dt_baseline(n, k, n_trails, trail_len, num_iters=100, mix_type=None, seed=None):
    mixture = create_mixture_dt(n, k, mix_type)# dt.Mixture.random(n, k)
    distribution = dt.Distribution.from_mixture(mixture, 3)
    sample = distribution.sample(n_trails * trail_len // 3)
    svd_learned_mixture = dt.svd_learn_new(sample, n, k)
    import htinf
    trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
    em_learned_mixture = dt.em_learn(n, k, np.array(trails) - 1, max_iter=num_iters)
    return {"svd_recovery_error": mixture.recovery_error(svd_learned_mixture),
            "em_recovery_error": mixture.recovery_error(em_learned_mixture)}


@genpath
def plot_test_mixture_dt(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    kwargs = {"mix_type": setup["mix_type"][0]} if "mix_type" in setup else {}
    df = df.join(df.astype("object").apply(lambda row: {
                      **test_mixture_dt(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, **kwargs, seed=row.seed),
                      # **test_mixture_dt_baseline(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, **kwargs, seed=row.seed)
                      },
                  axis=1, result_type='expand'))
    grp = df.groupby(["n"])

    # plt.figure(figsize=(6.1, 3.8))

    for grp_name, label in [("recovery_error", f"ULTRA-MC"),
                            ("svd_recovery_error", "SVD (discrete)"),
                            ("em_recovery_error", "EM (discrete)")]:
        x = grp[grp_name]
        mean = x.mean()
        std = x.std()
        config = next_config()
        plt.plot(mean.index, mean, label=label, **config)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.xticks(mean.index)
    plt.ylim(bottom=-0.007, top=0.257)
    plt.xlabel("number of states $n$")
    plt.ylabel("recovery error")
    plt.legend(loc="upper left")
    # plt.title(f"Learning {setup['k'][0]} chains from {setup['n_trails'][0]} trails of length {setup['trail_len'][0]} ({setup['mix_type'][0]})")
    savefig()


@memoize
def test_mixture_ct(n, k, n_trails, trail_len, num_iters=100, seed=None):
    mixture_dt = create_mixture_dt(n, k)
    mixture_dt.S[:] = 1
    mixture_dt.normalize()

    mixture_ct = ct.Mixture(mixture_dt.S, np.array([M - np.eye(n) for M in mixture_dt.Ms]))

    print(mixture_dt)
    print(mixture_ct)

    import htinf
    trails, _ = htinf.get_trails_ct(mixture_dt, n_trails, trail_len)

    """
    mixture = ct.Mixture.random(n, k)
    print(mixture)
    print(htinf.hitting_times_ct(mixture))
    """

    learned_mixture = htinf.em_ct(n, k, trails, num_iters=num_iters, learn_start=False)
    recovery_error = mixture_ct.recovery_error(learned_mixture)
    print(mixture_ct)
    print(learned_mixture)
    print("recovery_error=", recovery_error)
    # print(htinf.hitting_times_ct(mixture))
    # print(htinf.hitting_times_ct(learned_mixture))
    return {"recovery_error": recovery_error}


@memoize
def test_mixture_ct_baseline(n, k, n_trails, trail_len, num_iters=100, seed=None):
    return test_methods_with_baseline_ct(n, k, 0.1, trail_len, n_trails, seed=seed)

@genpath
def plot_test_mixture_ct(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: {
                      **test_mixture_ct(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, seed=row.seed),
                      # **test_mixture_ct_baseline(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, seed=row.seed)
                      },
                  axis=1, result_type='expand'))

    # import pdb; pdb.set_trace()
    print(df.columns)
    grp = df.groupby(["n"])
    # plt.figure(figsize=(6.1, 3.8))

    for grp_name, label in [("recovery_error", f"ULTRA-MC")]: # , ("continuous_em_recovery_error", "EM (continuous)"), ("kausik_recovery_error", "KTT (discretized)"), ("em_recovery_error", "EM (discretized)"), ("svd_recovery_error", "SVD (discretized)")]:
        print(grp_name)
        x = grp[grp_name]
        mean = x.mean()
        std = x.std()
        config = next_config()
        plt.plot(mean.index, mean, label=label, **config)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.ylim(bottom=-0.01, top=0.41)
    plt.xticks(mean.index)
    plt.xlabel("number of states $n$")
    plt.ylabel("recovery error")
    plt.legend(loc="upper right")
    # plt.title(f"Learning {setup['k'][0]} chains from {setup['n_trails'][0]} trails of length {setup['trail_len'][0]}")
    savefig()


def nba_print_mixture(mixture, state_dict, trails_ct_=None):
    k, n = mixture.S.shape

    def state_name(i):
        target_len = 12
        state = state_dict[i]
        if isinstance(state, tuple):
            s = (state[1] + "-" + state[0]).replace(' ', '')
        else:
            s = state
        s = s[:target_len-1] + "." if len(s) > target_len else s
        return s + (" " * (target_len - len(s)))

    trails_ct = [[(u-1, t) for u, t in trail] for trail in trails_ct_]
    htimes_chain = [[[] for _ in range(n)] for _ in range(k)]

    def stats(mixture, compute_htimes=True):
        log_ll = ct.likelihood(mixture, trails_ct)
        print("LL", np.max(np.exp(log_ll), axis=0))
        print("LL", np.sum(np.max(np.exp(log_ll), axis=0)))

        if compute_htimes:
            # cluster trails to compute avg holding time
            ll = np.exp(log_ll - np.max(log_ll, axis=0))
            ll /= np.sum(ll, axis=0)[None, :]

            clust = np.argmax(ll, axis=0)
            for i in range(k):
                htimes = htimes_chain[i]
                ixs, = np.where(clust == i)
                trails = [trails_ct[ix] for ix in ixs]
                for trail in trails:
                    for u, t in trail:
                        htimes[u].append(t)
                for u in range(n):
                    htimes[u] = np.mean(htimes[u])

        return np.sum(np.max(np.exp(log_ll), axis=0))

    def print_chain(S, K, T, l):
        starting_prob = np.sum(S)
        miss_prob, score_prob = (S @ T)[[0, 1]] / starting_prob
        print(f"miss={100*miss_prob:.2f}%, score={100*score_prob:.2f}% ({100*starting_prob:.2f}%)")
        if False: # trails_ct is not None:
            log_ll = ct.likelihood(mixture, trails_ct)
            ll = np.exp(log_ll - np.max(log_ll, axis=0))
            ll /= np.sum(ll, axis=0)[None, :]
            print("total likelihood:", np.sum(ll[l]))

        names = [state_name(i) for i in range(len(S))]
        print(" " * 14 + " ".join(names))
        with np.printoptions(precision=10, suppress=True, linewidth=np.inf):
            print(" " * 12, S)
        with np.printoptions(precision=9, suppress=True, linewidth=np.inf):
            print("\n".join([f"{name}{line}" for name, line in zip(names, str(K).split('\n'))]))
            """
            x = "Mixture(    # starting probabilities:\n "
            x += str(S).replace('\n', '\n ')
            x += "\n,           # rate matrices:\n "
            x += str(self.Ks).replace('\n', '\n ')
            x += "\n)"
            return x
            """

    our_ll = stats(mixture)

    mixture_stationary = mixture.Ts(10000)
    for l in range(mixture.L):
        print(f"\nChain {l}:")
        print_chain(mixture.S[l], mixture.Ks[l], mixture_stationary[l], l)

    real_players = ["PG", "SG", "PF", "SF", "C"]
    baskets = ["miss", "score"]
    # missing_ix = [i for i, x in state_dict.items() if x == "?"][0]
    for s, K, htimes in zip(mixture.S, mixture.Ks, htimes_chain):
        print("\n\n")
        for i1, p1 in state_dict.items():
            if p1 not in real_players: continue

            holding_time = htimes[i1]
            # holding_time = -1 / (K[i1,i1] + K[i1, missing_ix])
            print("    \\node[label={[label distance=6pt]" + ("below" if p1 in ["PF", "SF"] else "above") + ":{" + f"{holding_time:.1f}" + "s}}] at (" + p1 + ") {};")
            if s[i1] == max(s):
                print("    \\node[start] at (" + p1 + ") {};")
            for i2, p2 in state_dict.items():
                if p2 in real_players or p2 in baskets:
                    x = - K[i1, i2] / K[i1, i1]
                    if (p2 in real_players and x < 0.2) or (p2 in baskets and x < 0.15): continue
                    color = ("," + {"score": "green", "miss": "red"}[p2]) if p2 in baskets else ""
                    print("    \\draw[pass,opacity=" + str(x) + ",line width=" + str(x * 10) + "pt" + color + "] (" + p1 + ") to (" + p2 + ");")

    rnd_mixture = ct.Mixture.random(n, k)
    rnd_ll = stats(rnd_mixture, compute_htimes=False)
    return our_ll, rnd_ll

@memoize
def nba_learn_mixture(k, team, season, max_trail_time=20, min_trail_time=10, test_size=0.25, use_position=True, seed=0, num_iters=100, verbose=True):
    import NBA.extract as nba

    df, state_dict = nba.load_trails(team, season, tau=1, model_score=True, verbose=verbose, use_position=True, max_trail_time=max_trail_time, min_trail_time=min_trail_time)
    n = len(state_dict)
    # trails = [row.trail_ct + [(1 if row.ptsScored > 0 else 0, 1)] for _, row in df.iterrows()]
    trails = []
    for _, row in df.iterrows():
        trail = row.trail_ct
        final_state = 1 if row.ptsScored > 0 else 0
        last_player = np.random.randint(n) # trail[-1][0]
        trail = trail + [(final_state, 1), (last_player, 1)]
        trails.append([(u+1, t) for (u,t) in trail])

    # trails = train_iter.get_trails_ct()
    # print(trails)

    import htinf
    # H = htinf.hitting_times_from_trails_ct(n, trails)
    # print(H)

    mixture = htinf.em_ct(n, k, trails, num_iters=num_iters)
    return mixture, trails, state_dict


# @memoize
def nba_ht(k, team, season, max_trail_time=20, min_trail_time=10, test_size=0.25, use_position=True, seed=0, num_iters=100, verbose=True):

    # mixture = ct.Mixture.random(n, k)
    mixture, trails, state_dict = nba_learn_mixture(k, team, season,
            max_trail_time=max_trail_time, min_trail_time=min_trail_time,
            test_size=test_size, use_position=use_position, seed=seed,
            num_iters=num_iters, verbose=verbose)

    # import pdb; pdb.set_trace()

    our_ll, rnd_ll = nba_print_mixture(mixture, state_dict, trails)
    print("likelihood-ratio", our_ll / rnd_ll, our_ll, rnd_ll)



@memoize
def get_msnbc(num_categories):
    num_categories = 17
    n = num_categories

    all_trail_probs = np.zeros((n, n, n))
    num_visited = np.zeros(num_categories)
    trails = []

    print("reading file", end="")
    with open('msnbc990928.seq', 'r') as handle:
        lines = handle.readlines()
        for line_num, line in enumerate(lines):
            xs = [int(i)-1 for i in line.strip().split(" ")]
            for i in range(len(xs) // 3):
                x = xs[3*i:3*(i+1)]
                all_trail_probs[tuple(x)] += 1
                num_visited[x] += 1
            if len(xs) > 2 and len(set(xs)) > 1:
                trails.append(xs)
            if line_num % 10000 == 0: print(".", end="", flush=True)

    msnbc_distribution = dt.Distribution.from_all_trail_probs(all_trail_probs / np.sum(all_trail_probs))

    return msnbc_distribution, trails


@memoize
def test_msnbc_dt(n, k, learner, num_iters=100, seed=None):
    msnbc_distribution, trails = get_msnbc(n)
    print(f"obtained {len(trails)} trails of median length {np.median([len(trail) for trail in trails])}")

    if learner == "ht-em":
        import htinf
        trails_ = [[x+1 for x in trail] for trail in trails]
        learned_mixture = htinf.em(n, k, trails_, num_iters=num_iters)

    else:
        learned_mixture = dt.learners[learner](msnbc_distribution, n, k)

    learned_distribution = dt.Distribution.from_mixture(learned_mixture, 3)
    trail_error = msnbc_distribution.dist(learned_distribution)
    return {"trail_error": trail_error}

    """
    import htinf
    mixture = dt.Mixture.random(n, k)
    mixture.S[:] = 1
    mixture.normalize()
    trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
    learned_mixture = htinf.em(n, k, trails, num_iters=num_iters)
    recovery_error = mixture.recovery_error(learned_mixture)
    return {"recovery_error": recovery_error,
            **test_mixture_dt_baseline(n, k, n_trails, trail_len, num_iters=num_iters, seed=seed)}
    """

@genpath
def plot_test_msnbc_dt(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_msnbc_dt(row.n, row.k, row.learner, num_iters=row.num_iters, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby("learner")
    for learner, df in grp:
        grp = df.groupby("k")
        x = grp["trail_error"]
        mean = x.mean()
        std = x.std()

        config = next_config()
        plt.plot(mean.index, mean, label=learner, **config)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.xlabel("$L$")
    plt.ylabel("trail error")
    plt.legend(loc="upper left")
    plt.title(f"MSNBC")
    savefig()






def clustering_error(x, y, use_max=False):
    if use_max:
        n = len(x)
        x = np.argmax(x, axis=0)[None, :] == np.arange(n)[:, None]
        y = np.argmax(y, axis=0)[None, :] == np.arange(n)[:, None]
        # y = y == np.max(y, axis=0)[None, :]
        # x = x == np.max(x, axis=0)[None, :]
    dists = np.mean(np.abs(x.astype(float)[None, :, :] - y.astype(float)[:, None, :]), axis=2)
    row_ind, col_ind = linear_sum_assignment(dists)
    return np.sum(dists[row_ind, col_ind]) / 2

def gen_example(n, L, tau, t_len, n_samples):
    mixture = ct.Mixture.random(n, L)
    trails, chains = mixture.sample(n_samples=n_samples, t_len=t_len, tau=tau, return_chains=True)
    groundtruth = chains[None, :] == np.arange(L)[:, None]
    return mixture, trails, groundtruth

@memoize
def test_methods_ct(n, L, tau, t_len, n_samples, seed=None):
    print(">> gen_example")
    mixture, trails, groundtruth = gen_example(n, L, tau, t_len, n_samples)

    kausik_start_time = time.time()
    print(">> kausik_learn")
    if n < 8:
        kausik_mixture_ct, labels, kausik_mle_time = ct.kausik_learn(n, L, trails, tau, return_labels=True, return_time=True)
        kausik_time = time.time() - kausik_start_time
        kausik_lls = labels[None, :] == np.arange(L)[:, None]
        kausik_mixture_dt = dt.mle(n, trails, kausik_lls)
        kausik_recovery_error = mixture.recovery_error(kausik_mixture_ct)
        kausik_clustering_error = clustering_error(groundtruth, kausik_lls)
    else:
        kausik_mixture_ct = None
        kausik_mixture_dt = None
        kausik_lls = None
        kausik_recovery_error = None
        kausik_clustering_error = None
        kausik_time = None
        kausik_mle_time = None

    em_start_time = time.time()
    print(">> em_learn")
    em_mixture_dt = dt.em_learn(n, L, trails)
    em_lls = dt.likelihood(em_mixture_dt, trails)
    em_mle_start_time = time.time()
    em_mixture_ct = ct.mle_prior(em_lls, n, trails, tau=tau)
    em_mle_time = time.time() - em_mle_start_time
    em_time = time.time() - em_start_time

    svd_mixture_dt = None
    svd_mixture_ct = None
    svd_lls = None
    svd_time = None
    svd_mle_time = None

    print(">> Distribution.from_trails")
    sample = dt.Distribution.from_trails(n, trails)
    svd_start_time = time.time()
    print(">> svd_learn")
    svd_mixture_dt = dt.svd_learn(sample, n, L)
    svd_lls = dt.likelihood(svd_mixture_dt, trails)
    svd_mle_start_time = time.time()
    svd_mixture_ct = ct.mle_prior(svd_lls, n, trails, tau=tau)
    svd_mle_time = time.time() - svd_mle_start_time
    svd_time = time.time() - svd_start_time

    print("<< done")
    return {
        'mixture': mixture,
        'trails': trails,
        'groundtruth': groundtruth,

        'kausik_mixture_ct': kausik_mixture_ct,
        'kausik_mixture_dt': kausik_mixture_dt,
        'kausik_lls': kausik_lls,
        'kausik_recovery_error': kausik_recovery_error,
        'kausik_clustering_error': kausik_clustering_error,

        'em_mixture_ct': em_mixture_ct,
        'em_mixture_dt': em_mixture_dt,
        'em_lls': em_lls,
        'em_recovery_error': mixture.recovery_error(em_mixture_ct),
        'em_clustering_error': clustering_error(groundtruth, em_lls),

        'svd_mixture_ct': svd_mixture_ct,
        'svd_mixture_dt': svd_mixture_dt,
        'svd_lls': svd_lls,
        'svd_recovery_error': None if svd_mixture_ct is None else mixture.recovery_error(svd_mixture_ct),
        'svd_clustering_error': None if svd_lls is None else clustering_error(groundtruth, svd_lls),

        'kausik_time': kausik_time,
        'kausik_mle_time': kausik_mle_time,
        'em_time': em_time,
        'em_mle_time': em_mle_time,
        'svd_time': svd_time,
        'svd_mle_time': svd_mle_time,
    }

@memoize
def test_baseline_ct(n, L, tau, t_len, n_samples, seed=None):
    if n < 8:
        print(">> gen_example")
        mixture, trails, groundtruth = gen_example(n, L, tau, 1 + int(t_len), n_samples)
        duration = t_len * tau
        print(">> sample_ct")
        trails_ct = mixture.sample_ct(n_samples, duration)

        continuous_em_start_time = time.time()
        print(">> continuous_em_learn")
        continuous_em_mixture = ct.continuous_em_learn(n, L, trails_ct)
        print("<< done")
        continuous_em_time = time.time() - continuous_em_start_time
        continuous_em_lls = ct.likelihood(continuous_em_mixture, trails_ct)

        return {
            'mixture': mixture,
            'trails': trails,
            'groundtruth': groundtruth,

            'continuous_em_mixture': continuous_em_mixture,
            'continuous_em_lls': continuous_em_lls,
            'continuous_em_recovery_error': mixture.recovery_error(continuous_em_mixture),
            'continuous_em_clustering_error': clustering_error(groundtruth, continuous_em_lls),
            'continuous_em_time': continuous_em_time,
        }

    else:
        return {
            'mixture': None,
            'trails': None,
            'groundtruth': None,

            'continuous_em_mixture': None,
            'continuous_em_lls': None,
            'continuous_em_recovery_error': None,
            'continuous_em_clustering_error': None,
            'continuous_em_time': None,
        }

def test_methods_with_baseline_ct(*args, **kwargs):
    baseline = test_baseline_ct(*args, **kwargs)
    methods = test_methods_ct(*args, **kwargs)
    return {
        **baseline,
        **methods,
    }

    # (n, k, 0.1, trail_len, n_trails)

