#!/usr/local/bin/python3
""" Tests convergence of KL divergence (kNN) implementations with sample size"""
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def convergence_plot(estimator, test):
    """ Generates a plot of the convergence of an estimator with the sample
    size `N` in the case of a specific `test`.  Writes the plot to
    /figures/[test filename]_convergence.pdf and returns the filename."""

    k_values = [1, 5, 10]
    sample_sizes = [20, 50, 100, 200, 500, 1000]

    fig, ax = plt.subplots()

    for ik, k in enumerate(k_values):
        results = [test.compute(estimator, N, k) for N in sample_sizes]
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
    rootname = f'figures/{test.filename}_convergence'
    fig.savefig(f'{rootname}.png')
    fig.savefig(f'{rootname}.pdf')
    return f'{rootname}.png'

