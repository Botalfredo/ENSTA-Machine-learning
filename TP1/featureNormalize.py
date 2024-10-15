import numpy as np

def featureNormalize(X):
    """
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    # Compute the mean and standard deviation of each feature
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # Normalize each feature in X
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
