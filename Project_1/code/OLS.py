import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
    X = np.zeros((len(x), degree+1))
    for p in range(degree):
        X[:,p] = x**(p+1)
    return X


# Error computation
def R2(y_data,y_model):
    return 1 - np.sum ((y_data - y_model)**2)/np.sum((y_data - np.mean(y_data))** 2)

def MSE(y_data ,y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2)/n

# Creating lists to contain MSE and R2 values
MSE_list = []
R2_list = []
beta_values = []
degrees = np.arange(1, 6)

# Evaluating MSE and R2 for different design matrices
for i in degrees:
    X = design_matrix(x,i)
    # Splitting and scaling data
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2)

    X_train_mean = np.mean(X_train, axis=0)

    X_train_scaled =  X_train - X_train_mean
    X_test_scaled =  X_test - X_train_mean

    y_scaler = np.mean(y_train)
    y_train_scaled = y_train - y_scaler

    # SVD decomposition, full_matrices=false ensures that we only focus on non-zero singular values
    U, S, Vh = np.linalg.svd(X_train_scaled, full_matrices=False)

    #print(S)

    # Finding the pseudo-inverse of S
    S_inv = np.linalg.pinv(np.diag(S))

    #print(S_inv)

    # Calculating the beta values
    beta = Vh @ S_inv @ U.T @ y_train_scaled
    beta_values.append(beta)

    ytilde_OLS_train = X_train_scaled @ beta + y_scaler
    ytilde_OLS_test = X_test_scaled @ beta + y_scaler

    # Evaluating mean squared error (MSE) and R2
    MSE_list.append(MSE(y_train_scaled, ytilde_OLS_train))
    R2_list.append(R2(y_train_scaled, ytilde_OLS_train))

beta_values = beta_values[1:] # Deleting first element which is empty

print(MSE_list)
print(R2_list)
"""
plt.plot(degrees, MSE_list, label="MSE")
plt.legend()
plt.show()
plt.plot(degrees, R2_list, label="R2")
plt.legend()
plt.show()
"""


# Plotting beta values for each polynomial
#for i in range(1, len(beta_values)):
#   plt.plot(np.arange(1,i+1), beta_values[i])
#plt.show()

