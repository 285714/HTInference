# HTInference
Hitting Time Inference: Given a noisy subset of hitting times, we learn the underlying (discrete & continuous-time) Markov chain. We also implement EM to compute mixtures.
The Julia files contain only the optimized code, the Python files all previous versions, including naive derivation and numerical computation of the gradient.

- [Julia/ht-inf.jl] contains the Julia code to compute hitting time gradients, and perform gradient descent.
- [experiments.jl] and [scc_run.py] contain code to run experiments
