import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):

    # Reshape nn_params back into the parameters theta1 and theta2, the weight matrices
    # for our 2 layer neural network
    # Obtain theta1 and theta2 back from nn_params

    theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape((input_layer_size+1),hidden_layer_size).T
    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape((hidden_layer_size+1),num_labels).T

    # Setup some useful variables
    m, _ = X.shape

    # You need to return the following variables correctly
    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    # Add ones to the X data matrix

    # Add bias unit to a2
    X = np.concatenate([np.ones((np.shape(X)[0], 1)), X], axis=1)  # (5000, 26)

    # Construct a 10xm "y" matrix with all zeros and only one "1" entry
    # note here if the hand-written digit is "0", then that corresponds
    # to a y- vector with 1 in the 10th spot.
    y_matrix = np.zeros((num_labels, m))
    y_matrix[y.flatten(), np.arange(m)] = 1

    # Forward propagation
    a1 = X
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)

    # Add bias unit to a2
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)  # (5000, 26)

    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)

    # Compute the cost (without regularization)
    J = 1/m * np.sum(-y_matrix.T * np.log(a3) - (1 - y_matrix.T) * np.log(1 - a3))

    # Cost regularisation
    reg_term = (Lambda / (2 * m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
    J = J + reg_term

    # Gradients
    d3 = a3 - y_matrix.T
    d2 = theta2[:, 1:].T.dot(d3.T) * sigmoidGradient(z2.T)

    delta1 = d2.dot(a1)
    delta2 = d3.T.dot(a2)

    # Gradient regularisation
    theta1_grad = delta1 / m
    reg = (theta1[:, 1:] * Lambda) / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + reg

    theta2_grad = delta2 / m
    reg = (theta2[:, 1:] * Lambda) / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + reg

    # Unroll gradient
    grad = np.hstack((theta1_grad.T.ravel(), theta2_grad.T.ravel()))


    return J, grad