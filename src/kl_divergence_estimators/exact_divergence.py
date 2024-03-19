import numpy as np


def gaussian_divergence(mu1: float, mu2: float, sig1: float, sig2: float) -> float:
    """Analytical result for KL-divergence of two Gaussians.

    D( N(mu1, sig1) | N(mu2,sig2) )
    Ref: http://allisons.org/ll/MML/KL/Normal/
    """

    mudiff = pow(mu1 - mu2, 2)
    sigdiff = sig1 * sig1 - sig2 * sig2
    lograt = np.log(sig2 / sig1)
    return lograt + (mudiff + sigdiff) / (2 * sig2 * sig2)
