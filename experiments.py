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


def test_grid(n, max_iter):
    import htinf
    mixture = create_graph(n, "grid", selfloops=True)
    H_true = htinf.hitting_times(mixture)
    learned_mixture = htinf.ht_learn(H_true, max_iter=max_iter)
    tv = mixture.recovery_error(learned_mixture)
    print(mixture)
    print(learned_mixture)


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
def test_mixture_dt(n, k, n_trails, trail_len, num_iters=100, seed=None):
    import htinf
    mixture = dt.Mixture.random(n, k)
    mixture.S[:] = 1
    mixture.normalize()
    trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
    learned_mixture = htinf.em(n, k, trails, num_iters=num_iters)
    recovery_error = mixture.recovery_error(learned_mixture)
    return {"recovery_error": recovery_error,
            **test_mixture_dt_baseline(n, k, n_trails, trail_len, num_iters=num_iters, seed=seed)}


@memoize
def test_mixture_dt_baseline(n, k, n_trails, trail_len, num_iters=100, seed=None):
    mixture = dt.Mixture.random(n, k)
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
    df = df.join(df.astype("object").apply(lambda row: {
                      **test_mixture_dt(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, seed=row.seed),
                      **test_mixture_dt_baseline(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, seed=row.seed)},
                  axis=1, result_type='expand'))
    grp = df.groupby(["n"])

    for grp_name, label in [("recovery_error", f"EM HT {setup['num_iters'][0]} iterations"),
                            ("svd_recovery_error", "SVD"),
                            ("em_recovery_error", "EM")]:
        x = grp[grp_name]
        mean = x.mean()
        std = x.std()
        config = next_config()
        plt.plot(mean.index, mean, label=label, **config)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.xlabel("$n$")
    plt.ylabel("recovery error")
    plt.legend(loc="upper left")
    plt.title(f"Learning {setup['k'][0]} chains from {setup['n_trails'][0]} trails of length {setup['trail_len'][0]}")
    savefig()


@memoize
def test_mixture_ct(n, k, n_trails, trail_len, num_iters=100, seed=None):
    import htinf
    mixture = ct.Mixture.random(n, k)
    mixture.S[:] = 1
    mixture.normalize()
    trails, _ = htinf.get_trails_ct(mixture, n_trails, trail_len)
    print(trails)
    learned_mixture = htinf.em(n, k, trails, num_iters=num_iters)
    recovery_error = mixture.recovery_error(learned_mixture)
    return {"recovery_error": recovery_error}


@genpath
def plot_test_mixture_ct(setup, savefig=None):
    df = pd.DataFrame(itertools.product(*setup.values()), columns=setup.keys())
    df = df.join(df.astype("object").apply(lambda row: test_mixture_ct(row.n, row.k, row.n_trails, row.trail_len, num_iters=row.num_iters, seed=row.seed),
                  axis=1, result_type='expand'))

    grp = df.groupby(["n"])
    x = grp["recovery_error"]
    mean = x.mean()
    std = x.std()

    config = next_config()
    plt.plot(mean.index, mean, label=f"EM {setup['num_iters'][0]} iterations", **config)
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2)

    plt.xlabel("$n$")
    plt.ylabel("recovery error")
    plt.legend(loc="upper left")
    plt.title(f"Learning {setup['k'][0]} chains from {setup['n_trails'][0]} trails of length {setup['trail_len'][0]}")
    savefig()


def nba_print_mixture(mixture, state_dict, trails_ct=None):
    def state_name(i):
        target_len = 12
        state = state_dict[i]
        if isinstance(state, tuple):
            s = (state[1] + "-" + state[0]).replace(' ', '')
        else:
            s = state
        s = s[:target_len-1] + "." if len(s) > target_len else s
        return s + (" " * (target_len - len(s)))

    def print_chain(S, K, T, l):
        starting_prob = np.sum(S)
        miss_prob, score_prob = (S @ T)[[0, 1]] / starting_prob
        print(f"miss={100*miss_prob:.2f}%, score={100*score_prob:.2f}% ({100*starting_prob:.2f}%)")
        if trails_ct is not None:
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

    mixture_stationary = mixture.Ts(10000)
    for l in range(mixture.L):
        print(f"\nChain {l}:")
        print_chain(mixture.S[l], mixture.Ks[l], mixture_stationary[l], l)

    real_players = ["PG", "SG", "PF", "SF", "C"]
    baskets = ["miss", "score"]
    for s, K in zip(mixture.S, mixture.Ks):
        print("\n\n")
        for i1, p1 in state_dict.items():
            if p1 not in real_players: continue
            print("    \\node[label={[label distance=6pt]" + ("below" if p1 in ["PF", "SF"] else "above") + ":{" + f"{-10 * K[i1,i1]:.1f}" + "s}}] at (" + p1 + ") {};")
            if s[i1] == max(s):
                print("    \\node[start] at (" + p1 + ") {};")
            for i2, p2 in state_dict.items():
                if p2 in real_players or p2 in baskets:
                    x = - K[i1, i2] / K[i1, i1]
                    if (p2 in real_players and x < 0.2) or (p2 in baskets and x < 0.15): continue
                    color = ("," + {"score": "green", "miss": "red"}[p2]) if p2 in baskets else ""
                    print("    \\draw[pass,opacity=" + str(x) + ",line width=" + str(x * 10) + "pt" + color + "] (" + p1 + ") to (" + p2 + ");")


# @memoize
def nba_ht(k, team, season, max_trail_time=20, min_trail_time=10, test_size=0.25, use_position=True, seed=0, num_iters=100, verbose=True):
    """
    import NBA.learn as nba_learn

    data = lambda split: nba_learn.NBADataset(split, team, season, tau=0.1, max_trail_time=max_trail_time, min_trail_time=min_trail_time, test_size=test_size, use_position=use_position, seed=seed)
    train_iter, _ = data(split="train"), data(split="test")
    n = len(train_iter.state_dict)
    """

    """
    trails_dt = train_iter.get_trails_dt()
    mixture_dt = dt.em_learn(n, L, trails_dt, max_iter=1000)
    lls_dt = dt.likelihood(mixture_dt, trails_dt)
    mixture_ct = ct.mle_prior(lls_dt, n, trails_dt, tau=tau, max_iter=100)
    """

    import NBA.extract as nba

    df, state_dict = nba.load_trails(team, season, tau=1, model_score=True, verbose=verbose, use_position=True, max_trail_time=max_trail_time, min_trail_time=min_trail_time)
    n = len(state_dict)
    trails = [row.trail_ct + [(1 if row.ptsScored > 0 else 0, 1)] for _, row in df.iterrows()]
    trails = []
    for _, row in df.iterrows():
        trail = row.trail_ct
        final_state = 1 if row.ptsScored > 0 else 0
        last_player = trail[-1][0]
        trail = trail + [(final_state, 1), (last_player, 1)]
        trails.append([(u+1, t) for (u,t) in trail])

    # trails = train_iter.get_trails_ct()
    # print(trails)

    import htinf
    # H = htinf.hitting_times_from_trails_ct(n, trails)
    # print(H)

    mixture_dt = htinf.em_ct(n, k, trails, num_iters=num_iters)

    Ks = np.copy(mixture_dt.Ms)
    for i in range(k):
        Ks[i] -= np.eye(n)
    mixture = ct.Mixture(mixture_dt.S, Ks)
    print(mixture)

    # mixture = ct.Mixture.random(n, k)

    nba_print_mixture(mixture, state_dict)

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

