import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost for logistic regression with regularization

    Computes the cost of using theta as the parameter for regularized logistic regression.
    """

    # Initialize some useful values
    m, n = X.shape  # number of training examples and parameters
    theta = theta.reshape((n, 1))  # due to the use of fmin_tnc

    # Calculate the hypothesis
    h = sigmoid(X @ theta)

    # Compute the cost without regularization
    term1 = -y.T @ np.log(h)
    term2 = -(1 - y).T @ np.log(1 - h)
    cost = (term1 + term2) / m

    # Add regularization (but exclude theta_0)
    reg_term = (Lambda / (2 * m)) * np.sum(np.square(theta[1:]))

    # Total cost
    J = cost + reg_term

    return J