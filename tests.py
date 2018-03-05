#!/usr/local/bin/python3
""" Tests consistency and timing of KL divergence (kNN) implementations """
import timeit
import numpy as np

import knn_divergence as knn
import exact_divergence as exact
import universal_divergence as udv

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
print("KNN(naive): ", knn.naive_estimator(P1_1D, P2_1D, k))
print("KNN(skl):   ", knn.skl_estimator(P1_1D, P2_1D, k))
print("KNN(sci):   ", knn.scipy_estimator(P1_1D, P2_1D, k))
print("UDV(pip):   ", udv.estimate(P1_1D, P2_1D, k))

print(f"**** Self-divergence test (2D, N={sample_size}) ****")
print("Expectation: 0")
print("KNN(naive): ", knn.naive_estimator(P1_2D, P2_2D, k))
print("KNN(skl):   ", knn.skl_estimator(P1_2D, P2_2D, k))
print("KNN(sci):   ", knn.scipy_estimator(P1_2D, P2_2D, k))
print("UDV(pip):   ", udv.estimate(P1_2D, P2_2D, k))

print(f"**** Gaussian-divergence test (1D, N={sample_size}) ****")
print(f"Expectation: {exact.gaussian_divergence(P_mu,Q_mu,P_sig,Q_sig)}")
print("KNN(naive): ", knn.naive_estimator(P1_1D, Q1_1D, k))
print("KNN(skl):   ", knn.skl_estimator(P1_1D, Q1_1D, k))
print("KNN(sci):   ", knn.scipy_estimator(P1_1D, Q1_1D, k))
print("UDV(pip):   ", udv.estimate(P1_1D, Q1_1D, k))

print(f"**** Timing test (N={sample_size}) ****")
print(f"Naive: {timeit.timeit(lambda: knn.naive_estimator(P1_1D, P2_1D, k), number=2)}s")
print(f"SKL:   {timeit.timeit(lambda: knn.skl_estimator(P1_1D, P2_1D, k), number=2)}s")
print(f"Scipy: {timeit.timeit(lambda: knn.scipy_estimator(P1_1D, P2_1D, k), number=2)}s")
print(f"UDV:   {timeit.timeit(lambda: udv.estimate(P1_1D, P2_1D, k=k), number=2)}s")
print("Differences likely due to implementation rather than being intrinsic")
