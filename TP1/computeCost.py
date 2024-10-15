import numpy as np

def computeCost(X, y, theta):  
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    # m = number of training examples
    m = y.size

    # Hypothesis (predicted values)
    h = X.dot(theta)

    # Compute the squared errors
    squared_errors = (h - y) ** 2

    # Calculate the cost function
    J = (1 / (2 * m)) * np.sum(squared_errors)

    return J
