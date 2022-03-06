import numpy as np
import scipy.linalg


def normalize(X):
    x = X - np.mean(X, axis=0)
    # ddof is 'delta degrees of freedom', default is 0 (i.e. biased estimator)
    return x / np.std(x, axis=0, ddof=1)


def svd_of_covariance_matrix(X):
    _, num_features = X.shape

    # covariance matrix
    cov = (1 / num_features) * np.dot(X.T, X)

    U, s, Vh = scipy.linalg.svd(cov)
    S = scipy.linalg.diagsvd(s, num_features, num_features)

    return U, S


def project(X, U, projection_dimensions):
    """Project X onto principle components of U
    The first 'projection_dimentions' columns of U are used
    """
    return X.dot(U[:, :projection_dimensions])


def recover(X, U, projection_dimensions):
    """Recover data projected onto lower dimensional space"""
    return X.dot(U[:, :projection_dimensions].T)


def reduce_data(X, projection_dimensions):
    """Reduce vectors in X their component in the span of the first k eigenvectors
    where k is 'projection_dimensions'
    """
    U, _ = svd_of_covariance_matrix(X)

    V = U[:, :projection_dimensions]
    return X.dot(V).dot(V.T)
