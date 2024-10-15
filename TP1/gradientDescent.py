import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    theta, cost_history, theta_history = gradientDescent(X, y, theta, alpha, num_iters)
    Updates theta by taking num_iters gradient steps with learning rate alpha.
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    n = theta.size  # number of parameters
    cost_history = np.zeros(num_iters)  # cost over iterations
    theta_history = np.zeros((n, num_iters))  # theta over iterations

    for i in range(num_iters):
        # Compute the prediction error
        error = X.dot(theta) - y

        # Perform the gradient step for each theta parameter
        gradient = (1 / m) * X.T.dot(error)

        # Update theta
        theta = theta - alpha * gradient

        # Save the cost J in every iteration
        cost_history[i] = computeCost(X, y, theta)

        # Save the values of theta in every iteration
        theta_history[:, i] = theta.flatten()

    return theta, cost_history, theta_history