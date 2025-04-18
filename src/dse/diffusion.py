import numpy as np
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings("ignore")


def compute_diffusion_matrix(X: np.array, sigma: float = 10.0):
    '''
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X returns a diffusion matrix P, as an numpy ndarray.
    Using the "anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        sigma: a float
            conceptually, the neighborhood size of Gaussian kernel.
    Returns:
        K: a numpy array of size n x n that has the same eigenvalues as the diffusion matrix.
    '''

    # Construct the distance matrix.
    D = pairwise_distances(X)

    # Gaussian kernel
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))

    # Anisotropic density normalization.
    Deg = np.diag(1 / np.sum(G, axis=1)**0.5)
    K = Deg @ G @ Deg

    # Now K has the exact same eigenvalues as the diffusion matrix `P`
    # which is defined as `P = D^{-1} K`, with `D = np.diag(np.sum(K, axis=1))`.

    return K