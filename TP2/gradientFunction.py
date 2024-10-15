from sigmoid import sigmoid
import numpy as np


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic
    regression and the gradient of the cost w.r.t. to the parameters.
    """

    # Initialize some useful values
    # number of training examples
    m = X.shape[0]

    # number of parameters
    n = X.shape[1]
    theta = theta.reshape((n, 1)) # due to the use of fmin_tnc
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1/m) * np.dot(X.T, (h - y))

    return grad
