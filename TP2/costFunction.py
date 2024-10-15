import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression."""

    # Initialize some useful values
    m, n = X.shape  # number of training examples and parameters
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))

    return J
