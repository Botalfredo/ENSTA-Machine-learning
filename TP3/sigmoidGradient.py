from sigmoid import sigmoid

def sigmoidGradient(z):

    g = sigmoid(z) * ( 1 - sigmoid(z))

    return g
