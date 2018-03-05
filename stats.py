""" Basic statistical analysis of KL divergence estimators

    Provides a few (ducktyped) classes implementing analyses of KL-divergence estimators.
    Each analysis defines two probability distributions to be compared, and produces
    the KL-Divergence estimate along with an error estimate (the central 68% of values in
    `n_resample=100` resamples of the probability distributions).

"""
import logging as log
import numpy as np
from collections import namedtuple

from knn_divergence import naive_estimator as kl_divergence
from exact_divergence import gaussian_divergence
EstimatorStats = namedtuple("EstimatorStats", ["Mean", "Lower68", "Upper68", "MSE"])

n_resamples = 100
def divergence_estimate_analysis(P, Q, sample_size, k, expectation):
    """ Estimate the divergence D(P||Q) between samples P and Q of size `sample
    size` drawn from the provided probability distributions. Returns the mean,
    variance and MSE of `n_resample = 100` sample iterations"""
    log.debug(f"Divergence estimate: N = {sample_size}, iter = {n_resamples}")
    divergence_estimates = []
    for resample in range(0, n_resamples):
        P_sample = P(sample_size)
        Q_sample = Q(sample_size)
        divergence_estimates.append(kl_divergence(P_sample, Q_sample, k))
    divergence_estimates = np.sort(divergence_estimates)
    upper_limit = int(np.ceil( n_resamples*0.84))
    lower_limit = int(np.floor(n_resamples*0.16))

    Mean  = divergence_estimates.mean()
    Lower = Mean - divergence_estimates[lower_limit]
    Upper = divergence_estimates[upper_limit] - Mean
    MSE   = ((divergence_estimates - expectation) ** 2).mean()
    return EstimatorStats(Mean, Lower, Upper, MSE)


class self_divergence_estimate_1d:
    """ Estimate the divergence between two samples of size `sample size` and
    dimension 1, drawn from the same ~ N(0,1) probability distribution. Returns
    the mean, 68% confidence interval and MSE of an ensemble of resamples"""
    name  = "self_divergence_1d"
    title = "$\hat{D}_{\\mathrm{KL}}(P||P)$, $P \sim N(0,1)$"
    expectation = 0
    def P(self, sample_size):
        return np.random.multivariate_normal([0], [[1]], sample_size)
    def compute(self, sample_size, k):
        return divergence_estimate_analysis(self.P, self.P, sample_size, k, self.expectation)

class self_divergence_estimate_2d:
    """ Estimate the divergence between two samples of size `sample size` drawn
    from the same 2D distribution with
         mean=[0,0],  covariance=[[1, 0.1], [0.1, 1]]
    Returns the mean, 68% confidence interval and MSE of an ensemble of resamples"""
    name  = "self_divergence_2d"
    title = "$\hat{D}_{\\mathrm{KL}}(P||P)$, $P \sim N(\mathbf{0},[[1, 0.1],[0.1, 1]])$"
    expectation = 0
    def P(self, sample_size):
        CovMat = [[1, 0.1], [0.1, 1]]
        return np.random.multivariate_normal([0, 0], CovMat, sample_size)
    def compute(self, sample_size, k):
        return divergence_estimate_analysis(self.P, self.P, sample_size, k, self.expectation)

class gaussian_divergence_estimate_1d:
    """ Estimate the divergence between two samples of size `sample size` and
    dimension 1. The first drawn from N(0,1), the second from N(3,1). Returns
    the mean, 68% confidence interval and MSE of an ensemble of resamples"""
    name  = "gaussian_divergence_1d"
    title = "$\hat{D}_{\\mathrm{KL}}(P||Q)$, $P \sim N(0,1)$, $Q \sim N(3,1)$"
    expectation = gaussian_divergence(0, 3, 1, 1)
    def P(self, sample_size):
        return np.random.multivariate_normal([0], [[1]], sample_size)
    def Q(self, sample_size):
        return np.random.multivariate_normal([3], [[1]], sample_size)
    def compute(self, sample_size, k):
        return divergence_estimate_analysis(self.P, self.Q, sample_size, k, self.expectation)
