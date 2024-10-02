import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Ordinary least squared method

# Generating data to work with
noise = 0.1
x = np.linspace(0,1,100)
y = 2.0+5*x*x+noise*np.random.randn(100)

# For testing, the design matrix can be set to the identity matrix
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
MSE_test = mean_squared_error(y,ytilde_OLS)

# Testing MSE
assert (MSE_test == 0.0), "MSE test failed"

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

# Creating lists to contain MSE and R2 values
MSE = []
R2 = []
beta_values = []
degrees = np.linspace(1, 6, 7, dtype = int)
# Evaluating MSE and R2 for different design matrices
for i in degrees:
    X = design_matrix(x,i)
    # SVD decomposition, full_matrices=false ensures that we only focus on non-zero singular values
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    ytilde_OLS = (U @ U.T) @ y

    # Finding the pseudo-inverse of S
    S_inv = np.diag(1 / S)

    # Calculating the beta values
    beta = Vh @ S_inv @ U.T @ y
    beta_values.append(beta)

    # Evaluating mean squared error (MSE) and R2
    MSE.append(mean_squared_error(y, ytilde_OLS))
    R2.append(r2_score(y, ytilde_OLS))

print(MSE)
plt.plot(degrees, MSE, label="MSE")
plt.legend()
plt.show()
plt.plot(degrees, R2, label="R2")
plt.legend()
plt.show()

# Plotting beta values for 4th degree, need to plott for all polynomials
for i in range(1, len(degrees)):
    plt.plot(np.arange(0,i,1), beta_values[i])
plt.show()



# To visualize the regression, uncomment the lines below:
# plt.plot(x,y)
# plt.plot(x, X @ beta, '-')
# plt.show()
