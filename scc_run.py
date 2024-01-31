from experiments import *
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()



    """
    plot_test_mixture_dt({
        "n": np.linspace(3, 10, 9, dtype=int),
        "k": [2],
        "mix_type": ["stargrid"],
        "n_trails": [1000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(5, 10),
    })
    """


    """
    plot_test_mixture_ct({
        "n": np.linspace(2, 7, 6, dtype=int),
        "k": [2],
        "n_trails": [5000],
        "trail_len": [1000],
        "num_iters": [20],
        "seed": range(20, 25),
    })
    """

    plot_test_mixture_ct({
        "n": np.linspace(2, 10, 9, dtype=int),
        "k": [2],
        "n_trails": [5000],
        "trail_len": [1000],
        "num_iters": [20],
        "seed": range(20, 25),
    })

    """
    plot_test_mixture_ct({
        "n": np.linspace(2, 10, 9, dtype=int),
        "k": [2],
        "n_trails": [1000],
        "trail_len": [1000],
        "num_iters": [20],
        "seed": range(20, 25),
    })

    plot_test_mixture_ct({
        "n": np.linspace(2, 10, 9, dtype=int),
        "k": [2],
        "n_trails": [10000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(20, 25),
    })

    plot_test_mixture_ct({
        "n": np.linspace(2, 10, 9, dtype=int),
        "k": [2],
        "n_trails": [10000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(20, 25),
    })
    """



    # test_msnbc_dt(17, 5, "CA-SVD-EM2", num_iters=100, seed=None)

    """
    plot_test_msnbc_dt({
        "n": [17],
        "k": [3, 5, 7],
        "learner": ["CA-SVD-EM2", "ht-em"],
        "num_iters": [2],
        "seed": range(1),
    })
    """


    # test_grid(49, max_iter=100000)

    """
    plot_test_grid({
        "n": [25], # 16
        "max_iter": [100000],
#       "noise_std": [0, 0.01, 0.05, 0.1, 0.2],
        "noise_std": [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "seed": range(5),
    })
    """

    # print(test_runtime(5, max_iter=1000))

    """
    show_runtimes({
        "n": [5, 20, 50],
        "max_iter": [1000],
        "seed": range(5, 10),
    })
    """


    # nba_ht(2, "NYK", 2022, num_iters=100)


    """
    plot_test_single_chain({
        "n": np.linspace(20, 500, 5, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [10000],
        "trail_len": [10000],
        "max_iter": [10000],
        "seed": range(10, 11),
    })

    plot_test_single_chain({
        "n": np.linspace(20, 500, 5, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [10000],
        "trail_len": [10000],
        "max_iter": [50000],
        "seed": range(10, 11),
    })

    plot_test_single_chain({
        "n": np.linspace(20, 500, 10, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [10000],
        "trail_len": [10000],
        "max_iter": [10000],
        "seed": range(10, 11),
    })
    """

    """
    plot_test_single_chain({
        "n": np.linspace(20, 500, 10, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [10000],
        "trail_len": [10000],
        "max_iter": [10000],
        "seed": range(10, 15),
    })
    """


    """
    plot_test_single_chain({
        "n": [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], # np.linspace(20, 1000, 20, dtype=int),
        "graph_type": ["complete"],
        "random": [True],
        "from_trails": [False],
        "n_trails": [10000],
        "trail_len": [10000],
        "max_iter": [10000],
        "seed": range(15, 20),
    })
    """


    """
    plot_test_mixture_dt({
        "n": np.linspace(3, 10, 9, dtype=int),
        "k": [2],
        "n_trails": [1000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(5, 10),
    })

    plot_test_mixture_dt({
        "n": np.linspace(11, 20, 9, dtype=int),
        "k": [5],
        "n_trails": [5000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(5, 10),
    })
    """

    """
    plot_test_mixture_dt({
        "n": np.linspace(3, 10, 9, dtype=int),
        "k": [2],
        "n_trails": [30000],
        "trail_len": [100],
        "num_iters": [100],
        "seed": range(5, 10),
    })
    """



    """
    plot_test_single_chain({
        "n": np.linspace(4, 12, 9, dtype=int),
        "graph_type": ["complete", "lollipop", "star", "grid"],
        "random": [False],
        "from_trails": [False],
        "n_trails": [2000],
        "trail_len": [200],
        "max_iter": [10000],
        "seed": range(5),
    })

    plot_test_single_chain({
        "n": np.linspace(4, 12, 9, dtype=int),
        "graph_type": ["complete", "lollipop", "star", "grid"],
        "random": [False],
        "from_trails": [True],
        "n_trails": [2000],
        "trail_len": [200],
        "max_iter": [10000],
        "seed": range(5),
    })


    plot_test_single_chain({
        "n": np.linspace(5, 20, 10, dtype=int),
        "graph_type": ["complete", "lollipop", "star", "grid"],
        "random": [False],
        "from_trails": [False],
        "n_trails": [10000],
        "trail_len": [500],
        "max_iter": [50000],
        "seed": range(5),
    })

    plot_test_single_chain({
        "n": np.linspace(5, 20, 10, dtype=int),
        "graph_type": ["complete", "lollipop", "star", "grid"],
        "random": [False],
        "from_trails": [True],
        "n_trails": [10000],
        "trail_len": [500],
        "max_iter": [50000],
        "seed": range(5),
    })
    """








    """
    plot_test_mixture_ct({
        "n": np.linspace(2, 10, 10, dtype=int),
        "k": [2],
        "n_trails": [10000],
        "trail_len": [1000],
        "num_iters": [100],
        "seed": range(2),
    })
    """

    """
    plot_test_mixture({
        "n": np.linspace(10, 100, 10, dtype=int),
        "k": [2],
        "n_trails": [10000],
        "trail_len": [1000],
        "num_iters": [100],
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

