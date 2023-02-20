""" KL-Divergence estimation through K-Nearest Neighbours

    This module provides four implementations of the K-NN divergence estimator of
        Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú.
        "Divergence estimation for multidimensional densities via
        k-nearest-neighbor distances." Information Theory, IEEE Transactions on
        55.5 (2009): 2392-2405.

    The implementations are through:
        numpy (naive_estimator)
        scipy (scipy_estimator)
        scikit-learn (skl_estimator / skl_efficient)

    No guarantees are made w.r.t the efficiency of these implementations.

"""
import warnings

import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def knn_distance(point, sample, k):
    """Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample`

    This function works for points in arbitrary dimensional spaces.
    """
    # Compute all euclidean distances
    norms = np.linalg.norm(sample - point, axis=1)
    # Return the k-th nearest
    return np.sort(norms)[k]


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]


def naive_estimator(s1, s2, k=1):
    """KL-Divergence estimator using brute-force (numpy) k-NN
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    D = np.log(m / (n - 1))
    d = float(s1.shape[1])

    for p1 in s1:
        nu = knn_distance(p1, s2, k - 1)  # -1 because 'p1' is not in 's2'
        rho = knn_distance(p1, s1, k)
        D += (d / n) * np.log(nu / rho)
    return D


def scipy_estimator(s1, s2, k=1):
    """KL-Divergence estimator using scipy's KDTree
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d, nu_i = KDTree(s2).query(s1, k)
    rho_d, rhio_i = KDTree(s1).query(s1, k + 1)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
    else:
        D += (d / n) * np.sum(np.log(nu_d / rho_d[::, -1]))

    return D


def skl_estimator(s1, s2, k=1):
    """KL-Divergence estimator using scikit-learn's NearestNeighbours
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1).fit(s1)
    s2_neighbourhood = NearestNeighbors(n_neighbors=k).fit(s2)

    for p1 in s1:
        s1_distances, indices = s1_neighbourhood.kneighbors([p1], k + 1)
        s2_distances, indices = s2_neighbourhood.kneighbors([p1], k)
        rho = s1_distances[0][-1]
        nu = s2_distances[0][-1]
        D += (d / n) * np.log(nu / rho)
    return D


def skl_efficient(s1, s2, k=1):
    """An efficient version of the scikit-learn estimator by @LoryPack
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Contributed by Lorenzo Pacchiardi (@LoryPack)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
    s2_neighbourhood = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2)

    s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
    s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
    rho = s1_distances[:, -1]
    nu = s2_distances[:, -1]
    if np.any(rho == 0):
        warnings.warn(
            f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
            f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
            f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
            RuntimeWarning,
        )
    D = np.sum(np.log(nu / rho))

    return (d / n) * D + np.log(
        m / (n - 1)
    )  # this second term should be enough for it to be valid for m \neq n


# List of all estimators
Estimators = [naive_estimator, scipy_estimator, skl_estimator, skl_efficient]
