# ================ Introduction: load packages ================

# global
import numpy as np
import scipy.optimize as opt
import matplotlib as mpl
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt

# local
from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary,mapFeature
from costFunctionReg import costFunctionReg
from gradientFunctionReg import gradientFunctionReg
from predict import predict

# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     plotData.py
#     sigmoid.py
#     costFunction.py
#     gradientFunction.py
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).
path = 'ex2data2.txt'  
data = np.loadtxt(path, delimiter=',', dtype=float)


# set X (training data) and y (target variable)
nbCol = data.shape[1]  
X = data[:,0:nbCol-1]
y = data[:,nbCol-1:nbCol]



# tracer la figure
plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')


# =========== Part : Regularized Logistic Regression ============

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,0],X[:,1])


# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1],1))

# Set regularization parameter lambda to 1
Lambda = 1.0

# Compute and display initial cost and gradient for regularized logistic
# regression
cost= costFunctionReg(initial_theta, X, y, Lambda)
grad = gradientFunctionReg(initial_theta, X, y, Lambda)

print('\n -------------------------- \n')
print('Cost at initial theta (zeros): %f' % cost)
print('Expected cost (approx): 0.693147\n')
print('Gradient at initial theta (zeros) - first five values only: \n')
print(str(grad[0:5].T))
print('Expected gradients (approx) - first five values only:\n')
print(' 8.47457627e-03 1.87880932e-02 7.77711864e-05 5.03446395e-02 1.15013308e-02\n');


# Compute and display cost and gradient 
# with all-ones theta and lambda = 10

test_theta = np.ones((X.shape[1],1)) # np.atleast_2d(X.shape)
Lambda = 10.0

cost= costFunctionReg(test_theta, X, y, Lambda)
grad = gradientFunctionReg(test_theta, X, y, Lambda)

print('\n -------------------------- \n')
print('Cost at test theta (with lambda = 10): %f' % cost)
print('Expected cost (approx): 3.164509\n')
print('Gradient at test theta - first five values only: '+ str(grad[0:5]))
print('Expected gradients (approx) - first five values only:');
print(' 0.3460 0.1614 0.1948 0.2269 0.0922')


# ============= Part : Plotting the decision boundary =============

# Optimize and plot boundary

Lambda = 1.0

result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, fprime=gradientFunctionReg, args=(X, y,Lambda))  
cost = costFunctionReg(result[0], X, y,Lambda)  
theta = result[0]


# print theta to screen
print('\n -------------------------- \n')
print('Cost at theta found by fmin_tnc: %f' % cost)
print('Expected cost (approx): 0.5290')
print('\n -------------------------- \n')
print('theta:', ["%0.4f" % i for i in theta])
print('Expected theta (approx): 1.27 0.63 1.18 -2.02 -0.92')

# Plot Boundary
plotDecisionBoundary(theta, X, y, Lambda)


# Compute accuracy on our training set
p = predict(theta, X)
m = X.shape[0] 
accuracy = np.mean(np.double(p == np.squeeze(y))) * 100
print('\n -------------------------- \n')
print('Train Accuracy: %f' % accuracy)
print('Expected Accuracy (approx): 83.05 %')



## ============= Part 3: Optional Exercises =============


print('\n -------------------------- \n')
for Lambda in np.array([0,2,10,100]):
    theta = opt.fmin_tnc(costFunctionReg, initial_theta, fprime=gradientFunctionReg, args=(X, y,Lambda))  
    plotDecisionBoundary(theta[0], X, y, Lambda)

