#!/usr/local/bin/python3
""" Tests consistency and timing of KL divergence (kNN) implementations """
import knn_divergence as knn
import stats

k = 10
N = 1000

for test in stats.Tests:
    print(f'**** {test.name} ****')
    print(f' Expectation: {test.expectation}')
    for estimator in knn.Estimators:
        results = test.compute(estimator, N, k)
        print(f' {estimator.__name__}: {results.Mean} ({results.Time}s)')

print("Differences likely due to implementation rather than being intrinsic")
