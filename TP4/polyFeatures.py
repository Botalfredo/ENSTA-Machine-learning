import numpy as np

def polyFeatures(X, p):
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0], p))

    for i in range(1, p + 1):
        X_poly[:, i - 1] = X.flatten() ** i  # Fill the i-th column with X^i

    return X_poly
