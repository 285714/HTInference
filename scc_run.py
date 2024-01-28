from experiments import *
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()


    """
    nba_ht(2, "NYK", 2022)


    plot_test_mixture({
        "n": np.linspace(2, 20, 10, dtype=int),
        "k": [2],
        "n_trails": [10000],
        "trail_len": [100],
        "num_iters": [100],
        "seed": range(5),
    })

    plot_test_mixture({
        "n": np.linspace(10, 100, 10, dtype=int),
        "k": [2],
        "n_trails": [10000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(5),
    })
    """


    plot_test_single_chain({
        "n": np.linspace(20, 500, 10, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "max_iter": [50000],
        "seed": range(5),
    })


    """
    plot_test_single_chain({
        "n": np.linspace(5, 20, 5, dtype=int),
        "graph_type": ["complete", "lollipop", "star", "grid"],
        "random": [False],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "max_iter": [10000],
        "seed": range(5),
    })

    plot_test_single_chain({
        "n": np.linspace(5, 10, 6, dtype=int),
        "graph_type": ["complete", "lollipop", "star", "grid"],
        "random": [False],
        "from_trails": [True],
        "n_trails": [10000],
        "trail_len": [500],
        "max_iter": [10000],
        "seed": range(5),
    })
    """


    """
    plot_test_single_chain({
        "n": np.linspace(5, 20, 5, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "seed": range(5),
    })

    plot_test_single_chain({
        "n": np.linspace(5, 20, 5, dtype=int),
        "graph_type": ["lollipop"],
        "random": [False],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "seed": range(5),
    })

    plot_test_single_chain({
        "n": np.linspace(5, 20, 5, dtype=int),
        "graph_type": ["star"],
        "random": [False],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "seed": range(5),
    })


    plot_test_ht_sampling({
        "n": [10],
        "graph_type": ["complete"],
        "random": [True],
        "n_trails": np.linspace(100, 2000, 10, dtype=int),
        "trail_len": np.linspace(100, 200, 10, dtype=int),
        "seed": range(5),
    })

    plot_test_ht_sampling({
        "n": [10],
        "graph_type": ["lollipop"],
        "random": [False],
        "n_trails": np.linspace(100, 2000, 10, dtype=int),
        "trail_len": np.linspace(100, 200, 10, dtype=int),
        "seed": range(5),
    })

    plot_test_ht_sampling({
        "n": [10],
        "graph_type": ["star"],
        "random": [False],
        "n_trails": np.linspace(100, 2000, 10, dtype=int),
        "trail_len": np.linspace(100, 200, 10, dtype=int),
        "seed": range(5),
    })





    plot_test_single_chain({
        "n": np.linspace(20, 500, 10, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "seed": range(5),
    })
    """

