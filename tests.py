#!/usr/local/bin/python3
""" Tests consistency and timing of KL divergence (kNN) implementations """
import timeit
import numpy as np

from kldivergence import skl_knn_divergence, test_knn_divergence, gaussian_divergence, scipy_knn_divergence
from universal_divergence import estimate

k = 2
sample_size = 1000
P_mu, P_sig = 0, 1
Q_mu, Q_sig = 3, 1
P1_1D = np.random.multivariate_normal([P_mu], [[P_sig]], sample_size)
P2_1D = np.random.multivariate_normal([P_mu], [[P_sig]], sample_size)
Q1_1D = np.random.multivariate_normal([Q_mu], [[Q_sig]], sample_size)

CovMat = [[1, 0.1], [0.1, 1]]
P1_2D = np.random.multivariate_normal([0, 0], CovMat, sample_size)
P2_2D = np.random.multivariate_normal([0, 0], CovMat, sample_size)

print(f"**** Self-divergence test (1D, N={sample_size}) ****")
print("Expectation: 0")
print("KNN(naive): ", test_knn_divergence(P1_1D, P2_1D, k))
print("KNN(skl):   ", skl_knn_divergence(P1_1D, P2_1D, k))
print("KNN(sci):   ", scipy_knn_divergence(P1_1D, P2_1D, k))
print("UDV(pip):   ", estimate(P1_1D, P2_1D, k))

print(f"**** Self-divergence test (2D, N={sample_size}) ****")
print("Expectation: 0")
print("KNN(naive): ", test_knn_divergence(P1_2D, P2_2D, k))
print("KNN(skl):   ", skl_knn_divergence(P1_2D, P2_2D, k))
print("KNN(sci):   ", scipy_knn_divergence(P1_2D, P2_2D, k))
print("UDV(pip):   ", estimate(P1_2D, P2_2D, k))

print(f"**** Gaussian-divergence test (1D, N={sample_size}) ****")
print(f"Expectation: {gaussian_divergence(P_mu,Q_mu,P_sig,Q_sig)}")
print("KNN(naive): ", test_knn_divergence(P1_1D, Q1_1D, k))
print("KNN(skl):   ", skl_knn_divergence(P1_1D, Q1_1D, k))
print("KNN(sci):   ", scipy_knn_divergence(P1_1D, Q1_1D, k))
print("UDV(pip):   ", estimate(P1_1D, Q1_1D, k))

print(f"**** Timing test (N={sample_size}) ****")
print(f"Naive: {timeit.timeit(lambda: test_knn_divergence(P1_1D, P2_1D, k), number=2)}s")
print(f"SKL:   {timeit.timeit(lambda: skl_knn_divergence(P1_1D, P2_1D, k), number=2)}s")
print(f"Scipy: {timeit.timeit(lambda: scipy_knn_divergence(P1_1D, P2_1D, k), number=2)}s")
print(f"UDV:   {timeit.timeit(lambda: estimate(P1_1D, P2_1D, k=k), number=2)}s")
print("Differences likely due to implementation rather than being intrinsic")
