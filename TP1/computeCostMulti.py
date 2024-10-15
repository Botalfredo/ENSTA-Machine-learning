import numpy as np

def computeCostMulti(X, y, theta):  
    """
    Computes the cost of using theta as the parameter for linear
    regression to fit the data points in X and y.
    """
    m = y.size  # Number of training examples

    # Hypothesis (predicted values)
    h = X.dot(theta)

    # Compute the squared errors
    squared_errors = (h - y) ** 2

    # Calculate the cost function
    J = (1 / (2 * m)) * np.sum(squared_errors)

    return J
