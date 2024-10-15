import numpy as np
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the gradient of the cost w.r.t. to the parameters.
    """

    # Initialize some useful values
    m, n = X.shape  # number of training examples and parameters
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc

    # Calculate the hypothesis
    h = sigmoid(X @ theta)

    # Compute the gradient
    error = h - y

    # Compute the gradient for theta_0 (no regularization)
    grad = (X.T @ error) / m

    # Add regularization to the remaining theta terms (except theta_0)
    grad[1:] = grad[1:] + (Lambda / m) * theta[1:]

    return grad