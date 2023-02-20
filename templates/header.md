# KL divergence estimators

The [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
is normally defined between two probability distributions. In the case where
only samples of the probability distribution are available, the KL-divergence can
be estimated in a number of ways.

Here I test a few implementations of a KL-divergence estimator based on
k-Nearest-Neighbours probability density estimation.

The estimator is that of 

> Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú. 
> "Divergence estimation for multidimensional densities via k-nearest-neighbor distances." 
> Information Theory, IEEE Transactions on 55.5 (2009): 2392-2405.

Samples are drawn from various test distributions, and the estimated
KL-divergence between them is computed. Uncertainties are assessed by
re-sampling the distributions and re-computing divergence estimates 100 times.
Uncertainty bands are then given as the interval containing 68% of the
re-sampled estimates closest to the median. Timings where provided are the time
taken for the computation of all 100 re-samples on a sample size of **N**=1000
with **k**=5.

This study is far from exhaustive, and timings are sensitive to implementation
details. Please take with a pinch of salt.

