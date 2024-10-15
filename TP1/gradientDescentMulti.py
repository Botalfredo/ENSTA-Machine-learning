import numpy as np
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    theta, cost_history, theta_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
    Updates theta by taking num_iters gradient steps with learning rate alpha.
    """
    m = y.size  # Number of training examples
    n = theta.size  # Number of parameters
    cost_history = np.zeros(num_iters)
    theta_history = np.zeros((n, num_iters))

    for i in range(num_iters):
        # Compute the error (difference between prediction and actual values)
        error = X.dot(theta) - y

        # Update each parameter theta_j
        gradient = (1 / m) * X.T.dot(error)
        theta = theta - alpha * gradient

        # Save the cost J in every iteration
        cost_history[i] = computeCostMulti(X, y, theta)
        theta_history[:, i] = theta.flatten()

    return theta, cost_history, theta_history
