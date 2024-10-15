import numpy as np
from scipy.optimize import minimize

from linearRegCostFunction import linearRegCostFunction
from linearRegGradientFunction import linearRegGradientFunction


def trainLinearReg(X, y, Lambda, Xval=None, yval=None, maxiter=200):

    # Initialize Theta and costs history
    theta = np.zeros(X.shape[1])
    n = theta.size  # number of parameters
    cost_train = []
    cost_val = []

    thetas = []
    def save_step(*args):
        for arg in args:
            if type(arg) is np.ndarray:
                thetas.append(arg)

    initial_theta = np.zeros((X.shape[1]))

    costFunction = lambda t: linearRegCostFunction(X, y, t, Lambda)
    gradFunction = lambda t: linearRegGradientFunction(X, y, t, Lambda)

    result = minimize(costFunction, initial_theta, method='CG', jac=gradFunction, options={'disp': False, 'maxiter': maxiter, 'return_all':True}, callback=save_step)

    theta = result.x

    for t in thetas:
        cost_train.append(linearRegCostFunction(X, y, t, Lambda=0))

    if Xval is not None and yval is not None:
        for t in thetas:
            cost_val.append(linearRegCostFunction(Xval, yval, t, Lambda=0))

    # force to np array
    cost_train = np.array(cost_train)
    cost_val = np.array(cost_val)

    return theta, cost_train, cost_val
