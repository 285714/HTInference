from experiments import *
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()


    # test_single_chain(10, "complete", True)

    plot_test_single_chain({
        "n": np.linspace(5, 20, 5, dtype=int),
        "graph_type": ["complete", "lollipop", "star"],
        "random": [False],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "seed": range(5),
    })

    """
    plot_test_single_chain({
        "n": np.linspace(5, 20, 5, dtype=int),
        "graph_type": ["complete", "lollipop", "star"],
        "random": [True],
        "from_trails": [True],
        "n_trails": [2000],
        "trail_len": [200],
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

