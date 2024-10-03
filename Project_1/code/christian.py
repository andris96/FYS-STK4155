#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
# fig = plt.figure()
# ax = fig.add_subplot(projection = "3d")
# # Make data.
# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
# x, y = np.meshgrid(x,y)
# def FrankeFunction(x,y):
#     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
#     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
#     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
#     return term1 + term2 + term3 + term4

# z = FrankeFunction(x, y)
# # Plot the surface.
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
# linewidth=0, antialiased=False)
# # Customize the z axis.
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

#%%
from sklearn.model_selection import train_test_split
import numpy as np

def MSE(y, y_pred):
    return np.mean((y - y_pred)**2)

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
    beta = VT.T @ np.linalg.inv(np.diag(s)) @ U.T @ y
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

def scale_data(X):
    X_scaled = (X - np.mean(X))
    return X_scaled

order = 5
np.random.seed()
n = 100
method = "OLS"

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)
beta_buffer = np.zeros((order + 1, order+1))
MSE_buffer = np.zeros((order + 1))
R2_buffer = np.zeros((order + 1))
for o in range(order+1):
    X = create_design_matrix(x, o)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
    if method == "OLS":
        beta = OLS(X_train, y_train)
    elif method == "SVD":
        beta = SVD(X_train, y_train)
    y_pred = X_test @ beta
    if o == order+1:
        beta_buffer[o] = beta[:,0]
    else:
        beta_buffer[o, :o+1] = beta[:,0]
    MSE_buffer[o] = MSE(y_test, y_pred)
    R2_buffer[o] = R2(y_test, y_pred)
