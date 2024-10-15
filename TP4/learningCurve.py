import matplotlib.pyplot as plt
from trainLinearReg import trainLinearReg

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction
from linearRegGradientFunction import linearRegGradientFunction

def learningCurve(X, y, Lambda, Xval, yval, ax=plt.gca(), maxiter=2000):

    # Number of training examples
    m, _ = X.shape
    cost_train = 0.
    cost_val = 0.

    theta, cost_train, cost_val = trainLinearReg(X, y, Lambda, Xval=Xval, yval= yval, maxiter=maxiter)

    # =========================================================================

    return cost_train, cost_val