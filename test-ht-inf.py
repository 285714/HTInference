import numpy as np
import dtlearn as dt
import htinf
import experiments


n = 20
n_trails = 2000
trail_len = 200

mixture = experiments.create_graph(n, "lollipop")

trails, _ = htinf.get_trails(mixture, n_trails, trail_len)
H_sample = htinf.hitting_times_from_trails(n, trails)

learned_mixture = htinf.ht_learn(H_sample)
tv = mixture.recovery_error(learned_mixture)



