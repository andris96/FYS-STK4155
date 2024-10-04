#%%
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

def MSE(y, y_pred):
    n = np.size(y_pred)
    return np.sum((y - y_pred)**2)/n

def R2(y, y_pred):
    return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

def create_design_matrix(x, order):
    X = np.zeros((len(x), order+1))
    for i in range(order+1):
        X[:,i] = x[:,0]**i
    return X

def OLS(X, y):
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta

def SVD(X, y):
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    beta = VT.T @ np.linalg.pinv(np.diag(s)) @ U.T @ y
    return beta

def plot_beta(beta_buffer):
    beta_buffer[beta_buffer == 0] = np.nan

    for beta in beta_buffer:
        plt.plot(beta)
        plt.legend([f"Order {i}" for i in range(order+1)])
    plt.title("Beta coefficients")
    plt.xlabel("Order")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

def plot_MSE(MSE_buffer):
    plt.plot(MSE_buffer)
    plt.title("Mean Squared Error")
    plt.xlabel("Order")
    plt.ylabel("MSE")
    plt.grid()
    plt.show()

def plot_R2(R2_buffer):
    plt.plot(R2_buffer)
    plt.title("R2 score")
    plt.xlabel("Order")
    plt.ylabel("R2")
    plt.grid()
    plt.show()

# Standardized scaling should be working well, since the data
# should not contain many outliers as we are working with smooth functions
# with normalized noise
def scale_data(X):
    """
    Scales the features excluding the first column (which represents the intercept).
    """
    X_scaled = X.copy()
    mean = np.mean(X[:, 1:], axis=0)  # Exclude the first column (intercept)
    std = np.std(X[:, 1:], axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    # Scale the polynomial features (exclude the first column)
    X_scaled[:, 1:] = (X[:, 1:] - mean) / std
    return X_scaled, mean, std

order = 5
np.random.seed()
n = 100
method = "OLS"

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
# Buffer for storing results.
beta_buffer = np.zeros((order + 1, order+1))
MSE_buffer = np.zeros((order + 1))
R2_buffer = np.zeros((order + 1))
# Loop over polynomial orders.
for o in range(order+1):
    X = create_design_matrix(x, o)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)


    X_train_scaled, X_train_mean, X_train_std = scale_data(X_train)
    X_test_scaled = X_test.copy()
    X_test_scaled[:, 1:] = (X_test[:, 1:] - X_train_mean) / X_train_std


    # Target variable y is generally unecessary to scale for regression methods

    if method == "OLS":
        beta = OLS(X_train_scaled, y_train)
    elif method == "SVD":
        beta = SVD(X_train_scaled, y_train)

    y_pred = (X_test_scaled @ beta)


    beta_buffer[o, :o+1] = beta[:,0]
    MSE_buffer[o] = MSE(y_test, y_pred)
    R2_buffer[o] = R2(y_test, y_pred)


