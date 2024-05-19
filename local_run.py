from experiments import *
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # test_halyman(4, 2)

    """
    # test_agg(4, 1)

    nba_ht(6, "DEN", 2022, num_iters=100, max_trail_time=20)
    # nba_ht(3, "GSW", 2022, num_iters=100, max_trail_time=20)
    # nba_ht(3, "LAL", 2022, num_iters=100, max_trail_time=20)
    # nba_ht(3, "BOS", 2022, num_iters=100, max_trail_time=20)
    # nba_ht(3, "MIA", 2022, num_iters=100, max_trail_time=20)
    # nba_ht(3, "LAC", 2022, num_iters=100, max_trail_time=20)
    # nba_ht(3, "HOU", 2022, num_iters=100, max_trail_time=20)

    # ["GSW", "LAL", "BOS", "MIA", "LAC", "HOU"]
    """

    plot_test_grid({
        "n": [25],
        "max_iter": [100000],
        "noise_std": [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "noise": [1.0],
        "seed": range(5),
    })


    # test_wittmann(5, n_trails=10, t_len=1000)

    """
    plot_test_wittmann({
        "n": [5],
        "n_trails": [100],
        "t_len": np.linspace(10, 50, 10, dtype=int),
        "seed": range(5),
    })
    """


    # test_halyman(4, 2)

    # test_agg(4, 2)


    # nba_ht(6, "BOS", 2022, num_iters=100)

    """
    n = 5
    k = 2
    n_trails = 5000
    trail_len = 1000
    test_mixture_ct(n, k, n_trails, trail_len, num_iters=20)
    """


    """
    plot_test_mixture_dt({
        "n": np.linspace(3, 10, 10, dtype=int),
        "k": [2],
        "n_trails": [1000],
        "trail_len": [1000],
        "num_iters": [1000],
        "seed": range(10, 15),
    })
    """


    """
    f = 4

    plot_test_ht_sampling({
        "n": [16],
        "graph_type": ["complete"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(5, f*16, 10, dtype=int),
        "seed": range(10, 15),
    })

    plot_test_ht_sampling({
        "n": [16],
        "graph_type": ["lollipop"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(5, f*612, 10, dtype=int),
        "seed": range(10, 15),
    })

    plot_test_ht_sampling({
        "n": [16],
        "graph_type": ["star"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(100, f*46, 10, dtype=int),
        "seed": range(10, 15),
    })

    plot_test_ht_sampling({
        "n": [16],
        "graph_type": ["grid"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(100, f*60, 10, dtype=int), # 59.4
        "seed": range(10, 15),
    })
    """

