#!/usr/local/bin/python3
""" Tests convergence of KL divergence (kNN) implementations with sample size"""
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import stats

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

k_values = [1, 5, 10]
sample_sizes = [20, 50, 100, 200, 500, 1000]

tests = [stats.self_divergence_estimate_1d(),
         stats.self_divergence_estimate_2d(),
         stats.gaussian_divergence_estimate_1d()]

for test in tests:
    fig, ax = plt.subplots()

    self_divergence_vectorized = np.vectorize(test)
    for ik, k in enumerate(k_values):
        results = [test.compute(N, k) for N in sample_sizes]
        means   = [result.Mean for result in results]
        errup   = [result.Upper68 for result in results]
        errdn   = [result.Lower68 for result in results]
        xvalues = np.multiply(sample_sizes, np.exp((ik-1)/10))
        ax.errorbar(xvalues, means, errup, errdn, fmt='o', label=f'k={k}')

    # Axis formatting
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_xscale('log')
    ax.set_ylabel("Divergence Estimate")
    ax.set_xlabel("Sample size $(N)$")
    ax.axhline(y=test.expectation, color='black', linestyle='--', label='Expectation')

    # Legend
    legend = ax.legend(loc='best')
    legend.get_frame().set_alpha(0.8)

    ax.set_title(f"Convergence of {test.title}")
    fig.savefig(f'{test.name}_convergence.pdf')
