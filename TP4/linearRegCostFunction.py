import numpy as np

def linearRegCostFunction(X, y, theta, Lambda):

    m,n = X.shape # number of training examples
    theta = theta.reshape((n,1)) # in case where theta is a vector (n,) 
    J = 0.

    y = y.reshape((m, 1))
    h = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(h - y))
    reg_term = (Lambda / (2 * m)) * np.sum(np.square(theta[1:]))
    J = cost + reg_term

    return J.flatten()







