import numpy as np

from sigmoid import sigmoid

def predictNeuralNetwork(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

    m, _ = X.shape
    num_labels, _ = Theta2.shape

    z2 = X @ Theta1.T
    a2 = sigmoid(z2)

    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    p = np.argmax(a3, axis=1)
    
    return p

