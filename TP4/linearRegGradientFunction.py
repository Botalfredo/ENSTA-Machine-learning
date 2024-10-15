import numpy as np

def linearRegGradientFunction(X, y, theta, Lambda):
    # Initialize some values
    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,)
    grad = 0.

    y = y.reshape((m, 1))
    h = X.dot(theta)
    grad = (1 / m) * (X.T.dot(h - y))

    reg_term = (Lambda / m) * theta
    reg_term[0] = 0

    grad = grad + reg_term

    return grad.flatten()
