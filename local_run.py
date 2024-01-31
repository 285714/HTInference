from experiments import *
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()

    nba_ht(6, "BOS", 2022, num_iters=100)

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

    plot_test_ht_sampling({ # CT = 10
        "n": [10],
        "graph_type": ["complete"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(5, f*10, 10, dtype=int),
        "seed": range(10, 15),
    })

    plot_test_ht_sampling({ # CT = 165
        "n": [10],
        "graph_type": ["lollipop"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(5, f*165, 10, dtype=int),
        "seed": range(5, 10),
    })

    plot_test_ht_sampling({ # CT = 28
        "n": [10],
        "graph_type": ["star"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(100, f*28, 10, dtype=int),
        "seed": range(5, 10),
    })

    plot_test_ht_sampling({ # CT = 40
        "n": [10],
        "graph_type": ["grid"],
        "random": [False],
        "n_trails": np.linspace(10, 100, 10, dtype=int),
        "trail_len": np.linspace(100, f*40, 10, dtype=int),
        "seed": range(5, 10),
    })
    """

