# HTInference
Hitting Time Inference: Given a noisy subset of hitting times, we learn the underlying (discrete & continuous-time) Markov chain. We also implement EM to compute mixtures.
The Julia files contain only the optimized code, the Python files all previous versions, including naive derivation and numerical computation of the gradient.

- [Julia/ht-inf.jl](Julia/ht-inf.jl) contains the Julia code to compute hitting time gradients, and perform gradient descent.
- [experiments.py](experiments.py) and [scc_run.py](scc_run.py) contain code to run experiments

Below is an example use:

```python
import dtlearn as dt
import htinf

# create a single random Markov chain (a mixture containing only a single chain)
n = 10
mixture = dt.Mixture.random(n, 1)

# obtain the hitting times
H_true = htinf.hitting_times(mixture)
learned_mixture = htinf.ht_learn(H_true)

# report error
tv = mixture.recovery_error(learned_mixture)
print(tv)
```
