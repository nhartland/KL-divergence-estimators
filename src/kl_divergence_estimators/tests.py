""" Basic statistical analysis of KL divergence estimators

    Provides a few (ducktyped) classes implementing analyses of KL-divergence estimators.
    Each analysis defines two probability distributions to be compared, and produces
    the KL-Divergence estimate along with an error estimate (the central 68% of values in
    `n_resamples=100` resamples of the probability distributions).

"""
import time
import numpy as np
import logging as log
from collections import namedtuple

from exact_divergence import gaussian_divergence

EstimatorStats = namedtuple(
    "EstimatorStats", ["Estimator", "Mean", "Lower68", "Upper68", "MSE", "Time"]
)


def divergence_estimate_analysis(
    estimator, P, Q, sample_size, k, expectation, n_resamples=100
):
    """Estimate the divergence D(P||Q) between samples of size `sample size`
    drawn from the provided probability distributions `P` and `Q`.  Returns the
    mean, variance and MSE of `n_resample = 100` sample iterations.

    NOTE: For consistency this function re-seeds the numpy RNG to ensure that
    samples are identical between estimators

    Args:
        estimator (Callable): The KLD estimator to be tested
        P (Callable): A callback that generates samples of `P`
        Q (Callable): A callback that generates samples of `Q`
        sample_size (int): The number of samples to draw from `P` and `Q`
        k (int): The number of nearest neighbours to use in the estimation
        expectation (float): The true value of the KLD (for computing the MSE)
        n_resamples (int): Number of different samples of `P` and `Q` for computing
                           the confidence interval of the estimator.
    """
    np.random.seed(0)  # Reseed the RNG
    start_time = time.time()
    log.info(
        f" Running test with {estimator.__name__}: N = {sample_size}, iter = {n_resamples}"
    )
    divergence_estimates = []

    # Repeatedly resample and compute the divergence
    for resample in range(0, n_resamples):
        P_sample = P(sample_size)
        Q_sample = Q(sample_size)
        divergence_estimates.append(estimator(P_sample, Q_sample, k))

    # Compute the 68% confidence interval of the estimator
    divergence_estimates = np.sort(divergence_estimates)
    upper_limit = int(np.ceil(n_resamples * 0.84))
    lower_limit = int(np.floor(n_resamples * 0.16))

    # Summary statistics of the estimator
    Mean = divergence_estimates.mean()
    Lower = Mean - divergence_estimates[lower_limit]
    Upper = divergence_estimates[upper_limit] - Mean
    MSE = ((divergence_estimates - expectation) ** 2).mean()

    return EstimatorStats(
        estimator.__name__, Mean, Lower, Upper, MSE, time.time() - start_time
    )


class self_divergence_estimate_1d:
    """Estimate the divergence between two samples of size **N** and dimension
    1, drawn from the same ~ N(0,1) probability distribution."""

    name = "Self-divergence of samples from a 1-dimensional Gaussian"
    filename = "self_divergence_1d"
    title = "$\hat{D}_{\\mathrm{KL}}(P||P)$, $P \sim N(0,1)$"
    expectation = 0

    def P(self, N):
        return np.random.multivariate_normal([0], [[1]], N)

    def compute(self, estimator, N, k):
        return divergence_estimate_analysis(
            estimator, self.P, self.P, N, k, self.expectation
        )


class self_divergence_estimate_2d:
    """Estimate the divergence between two samples of size **N** drawn
    from the same 2D distribution with
    `mean=[0,0]` and `covariance=[[1, 0.1], [0.1, 1]]`."""

    name = "Self-divergence of samples from a 2-dimensional Gaussian"
    filename = "self_divergence_2d"
    title = "$\hat{D}_{\\mathrm{KL}}(P||P)$, $P \sim N(\mathbf{0},[[1, 0.1],[0.1, 1]])$"
    expectation = 0

    def P(self, N):
        CovMat = [[1, 0.1], [0.1, 1]]
        return np.random.multivariate_normal([0, 0], CovMat, N)

    def compute(self, estimator, N, k):
        return divergence_estimate_analysis(
            estimator, self.P, self.P, N, k, self.expectation
        )


class gaussian_divergence_estimate_1d:
    """Estimate the divergence between two samples of size `N` and dimension
    1. The first drawn from N(0,1), the second from N(2,1)."""

    name = "Divergence of two 1-dimensional Gaussians"
    filename = "gaussian_divergence_1d"
    title = "$\hat{D}_{\\mathrm{KL}}(P||Q)$, $P \sim N(0,1)$, $Q \sim N(2,1)$"
    expectation = gaussian_divergence(0, 2, 1, 1)

    def P(self, N):
        return np.random.multivariate_normal([0], [[1]], N)

    def Q(self, N):
        return np.random.multivariate_normal([2], [[1]], N)

    def compute(self, estimator, N, k):
        return divergence_estimate_analysis(
            estimator, self.P, self.Q, N, k, self.expectation
        )


# List of implemented tests
Tests = [
    self_divergence_estimate_1d(),
    self_divergence_estimate_2d(),
    gaussian_divergence_estimate_1d(),
]
