#%%
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold


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

# Creates a design matrix for 2D input (x, y) up to a given polynomial order.
def create_design_matrix_2d(x, y, order):
    l = int((order + 1)*(order + 2)/2)  # Number of polynomial terms
    X = np.ones((len(x), l))

    idx = 1
    for i in range(1, order + 1):
        for j in range(i + 1):
            X[:, idx] = (x**(i - j)) * (y**j)
            idx += 1
    return X


def OLS(X, y):
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta

def SVD(X, y):
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    beta = VT.T @ np.linalg.pinv(np.diag(s)) @ U.T @ y
    return beta


def compute_beta_ridge_svd(X, y, lambda_):
    U, s, VT = np.linalg.svd(X, full_matrices=False)

    S_diag = np.diag(s)
    S_squared = S_diag.T @ S_diag
    lambda_I = lambda_ * np.eye(S_squared.shape[0])
    S_inv = np.linalg.pinv(S_squared + lambda_I)
    beta_ridge = VT.T @ S_inv @ S_diag.T @ U.T @ y

    return beta_ridge

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

def plot_test_vs_train(MSE_buffer_test, MSE_buffer_train):
    plt.plot(MSE_buffer_test, label="Test")
    plt.plot(MSE_buffer_train, label="Train")
    plt.title("Mean Squared Error")
    plt.xlabel("Complexity")
    plt.ylabel("MSE")
    plt.legend()
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

def FrankeFunction(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        noise = np.random.normal(0, 0.1, x.shape)
        return term1 + term2 + term3 + term4 + noise

def kfold(X_train, y_train, k, method = "OLS", lambda_ = 0.001):
    kfold = KFold(n_splits=k).split(X_train_scaled, y_train)
    scores = []

    for train_idx, test_idx in kfold:
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

        X_fold_train_scaled, train_mean, train_std = scale_data(X_fold_train)
        X_fold_test_scaled = X_fold_test.copy()
        X_fold_test_scaled[:, 1:] = (X_fold_test[:, 1:] - train_mean) / train_std

        if method == "OLS":
            beta = OLS(X_fold_train_scaled, y_fold_train)
            y_pred = X_fold_test_scaled @ beta
            scores.append(MSE(y_fold_test, y_pred))

        if method == "ridge":
            beta = compute_beta_ridge_svd(X_fold_train_scaled, y_fold_train, lambda_)
            y_pred = X_fold_test_scaled @ beta
            scores.append(MSE(y_fold_test, y_pred))

        if method == "lasso":
            RegLasso = Lasso(alpha=lambda_, fit_intercept=False, max_iter=10000)
            RegLasso.fit(X_fold_train_scaled, y_fold_train)
            y_pred = RegLasso.predict(X_fold_test_scaled)
            scores.append(MSE(y_fold_test, y_pred))

    return np.mean(scores)


#%%

k = 10
order = 5
np.random.seed()
n = 500
# Adjust the method and data to test different methods and data sets
method = "SVD"
lambdas =[0.001, 0.01, 0.1, 1.0]
data = "2D"

# Make data set with either 1D function or 2D function
if data == "1D":
    x = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

elif data == "2D":
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)
    z = FrankeFunction(x, y)

    # Flatten the data for regression
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()


#%

# Buffer for storing results.
if data == "1D":
    beta_buffer = np.zeros((order + 1, order+1))
elif data == "2D":
    beta_buffer = np.zeros((order + 1, int((order + 1)*(order + 2)/2)))


MSE_buffer_test = np.zeros((order + 1))
MSE_buffer_train = np.zeros((order + 1))
MSE_buffer_ridge = np.zeros(((order + 1), len(lambdas)))
MSE_buffer_lasso = np.zeros(((order + 1), len(lambdas)))
R2_buffer = np.zeros((order + 1))

MSE_buffer_kfold_OLS = np.zeros(((order + 1), len(lambdas)))
MSE_buffer_kfold_ridge = np.zeros(((order + 1), len(lambdas)))
MSE_buffer_kfold_lasso = np.zeros(((order + 1), len(lambdas)))



# Loop over polynomial orders.
for o in range(order+1):
    # Create design matrix for 1D or 2D data
    if data == "1D":
        X = create_design_matrix(x, o)
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
    elif data == "2D":
        X = create_design_matrix_2d(x, y, o)
        # note that we are using z as the target variable here
        # y_train and y_test could be called z_train and z_test, but we keep the names for simplicity
        (X_train, X_test, y_train, y_test) = train_test_split(X, z, test_size=0.2)


    X_train_scaled, X_train_mean, X_train_std = scale_data(X_train)
    X_test_scaled = X_test.copy()
    X_test_scaled[:, 1:] = (X_test[:, 1:] - X_train_mean) / X_train_std




    # Target variable y is generally unecessary to scale for regression methods

    if method == "OLS":
        beta = OLS(X_train_scaled, y_train)
    elif method == "SVD":
        beta = SVD(X_train_scaled, y_train)


    beta_ridge_svd = compute_beta_ridge_svd(X_train_scaled, y_train, lambdas[1])


    if data == "1D":
        beta_buffer[o, :o+1] = beta[:,0]

    elif data == "2D":
        num_coeff = int((o + 1)*(o + 2)/2)
        beta_buffer[o, :num_coeff] = beta

     # OLS
    y_pred = (X_test_scaled @ beta)
    y_pred_train = (X_train_scaled @ beta)

    # Store the results
    MSE_buffer_test[o] = MSE(y_test, y_pred)
    MSE_buffer_train[o] = MSE(y_train, y_pred_train)
    R2_buffer[o] = R2(y_test, y_pred)

    # Loop over lambdas for Ridge and Lasso
    for i in range(len(lambdas)):
        # Ridge
        beta_ridge_svd = compute_beta_ridge_svd(X_train_scaled, y_train, lambdas[i])
        y_pred_ridge = (X_test_scaled @ beta_ridge_svd)

        # Lasso
        RegLasso = Lasso(alpha=lambdas[i], fit_intercept=False, max_iter=10000)
        RegLasso.fit(X_train_scaled, y_train)
        y_pred_lasso = RegLasso.predict(X_test_scaled)

        # Store the results
        MSE_buffer_ridge[o, i] = MSE(y_test, y_pred_ridge)
        MSE_buffer_lasso[o, i] = MSE(y_test, y_pred_lasso)
        MSE_buffer_kfold_OLS[o, i] = kfold(X_train_scaled, y_train, k, "OLS", lambdas[i])
        MSE_buffer_kfold_ridge[o, i] = kfold(X_train_scaled, y_train, k, "ridge", lambdas[i])
        MSE_buffer_kfold_lasso[o, i] = kfold(X_train_scaled, y_train, k, "lasso", lambdas[i])


    MSE_buffer_test[o] = MSE(y_test, y_pred)
    MSE_buffer_train[o] = MSE(y_train, y_pred_train)
    R2_buffer[o] = R2(y_test, y_pred)

#%%
