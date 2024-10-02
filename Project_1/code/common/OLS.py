import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Ordinary least squared method

# Generating data to work with
noise = 0.1
x = np.linspace(0,1,100)
y = 2.0+5*x*x+noise*np.random.randn(100)

# Design matrix is set to the identity matrix
# By doing this we ensure that the MSE should evaluate to 0
X = np.eye(len(x))

"""
We should use SVD to attain y_tilde. Matrix inversion cannot be done on defective matrices,
but SVD can be done on any matrix. Thus we avoid any potential problems to do with computing
the inverse of (X^T X). It requires more computation, but that is no problem in this case.
"""
U, S, Vh = np.linalg.svd(X)
ytilde_OLS = (U @ U.T) @ y

# Evaluating mean squared error (MSE)
MSE = mean_squared_error(y,ytilde_OLS)

print(MSE)
"""
 We will now work with polynomial fitting, and compare the regression analysis
 depending on the degree of the polynomial. First we need to attain the design matrix.
 We do this by having a function which takes x and the degree of our polynomial as input,
 and returns the design matrix as output.
"""
def design_matrix(x, degree):
    X = np.zeros((len(x), degree))
    for p in range(degree):
        X[:,p] = x**p
    return X

X = design_matrix(x, 5)

# SVD decomposition, full_matrices= false ensures that we only focus on non-zero singular values
U, S, Vh = np.linalg.svd(X, full_matrices=False)
ytilde_OLS = (U @ U.T) @ y

# Finding the pseudo-inverse of S
S_inv = np.diag(1 / S)

# Calculating the beta values
beta = Vh @ S_inv @ U.T @ y

# Evaluating mean squared error (MSE)
MSE = mean_squared_error(y,ytilde_OLS)

print(MSE)

plt.plot(x,y)
plt.plot(x, X @ beta, '-')
plt.show()

