import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from utils.franke import FrankeFunction
import seaborn as sns
from utils.regression_functions import OLS_GradientDescent, OLS, Ridge, Ridge_GradientDescent, OLS_StochasticGradientDescent, Ridge_StochasticGradientDescent
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.optimizers import find_optimal_eta_OLS_GD, find_optimal_eta_Ridge_GD, find_optimal_eta_batch_OLS_SGD, find_optimal_eta_batch_Ridge_SGD

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def plot_convergence_rate(OLS_GD, Ridge_GD, OLS_GD_momentum, Ridge_GD_momentum, OLS_SGD, Ridge_SGD, OLS_SGD_momentum, Ridge_SGD_momentum):
    plt.figure()
    plt.plot(OLS_GD.mse_history, label="OLS_GD")
    plt.plot(Ridge_GD.mse_history, label="Ridge_GD", linestyle="--")
    plt.plot(OLS_GD_momentum.mse_history, label="OLS_GD with momentum")
    plt.plot(Ridge_GD_momentum.mse_history, label="Ridge_GD with momentum", linestyle="--")
    plt.plot(OLS_SGD.mse_history, label="OLS_SGD")
    plt.plot(Ridge_SGD.mse_history, label="Ridge_SGD", linestyle="--")
    plt.plot(OLS_SGD_momentum.mse_history, label="OLS_SGD with momentum")
    plt.plot(Ridge_SGD_momentum.mse_history, label="Ridge_SGD with momentum", linestyle="--")
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.title("Convergence rate of Gradient Descent methods", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.show()

def compare_adagrad(OLS_GD, OLS_GD_momentum, OLS_SGD, OLS_SGD_momentum, OLS_Adagrad, OLS_Adagrad_momentum, OLS_SGD_adagrad, OLS_SGD_adagrad_momentum):
    plt.figure()
    plt.plot(OLS_GD.mse_history, label="OLS_GD")
    plt.plot(OLS_GD_momentum.mse_history, label="OLS_GD with momentum")
    plt.plot(OLS_SGD.mse_history, label="OLS_SGD")
    plt.plot(OLS_SGD_momentum.mse_history, label="OLS_SGD with momentum")
    plt.plot(OLS_Adagrad.mse_history, label="OLS_Adagrad", linestyle="--")
    plt.plot(OLS_Adagrad_momentum.mse_history, label="OLS_Adagrad with momentum", linestyle="--")
    plt.plot(OLS_SGD_adagrad.mse_history, label="OLS_SGD_Adagrad", linestyle="--")
    plt.plot(OLS_SGD_adagrad_momentum.mse_history, label="OLS_SGD_Adagrad with momentum", linestyle="--")
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.title("Convergence rate of OLS with Adagrad", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.show()

def main():
    franke = FrankeFunction(eps=0.1, n=500)
    x_franke, y_franke = franke.generate_data()
    X_franke = franke.create_design_matrix(degree=5)
    # franke.plot_franke()

    X_train, X_test, y_train, y_test = train_test_split(X_franke, y_franke, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train, X_test = scale_data(X_train, X_test)
    X_train, X_val = scale_data(X_train, X_val)

    OLS_GD_lr = 0.01
    OLS_GD_momentum_lr = 0.01
    Ridge_GD_lr = 0.01
    Ridge_GD_lambda = 0.1
    Ridge_GD_momentum_lr = 0.1
    Ridge_GD_momentum_lambda = 0.0001

    OLS_SGD_lr = 0.01
    OLS_SGD_batch_size = 1000

    Ridge_SGD_lr = 0.1
    Ridge_SGD_lambda = 0.001
    Ridge_SGD_batch_size = 100

    OLS_SGD_adagrad_momentum = OLS_StochasticGradientDescent(lr=OLS_SGD_lr, batch_size=128, momentum=0.5, autograd=True, adagrad=True)
    OLS_SGD_adagrad_momentum.fit(X_train, y_train, X_val, y_val)
    y_pred = OLS_SGD_adagrad_momentum.predict(X_test)

    print(f"MSE: {np.mean((y_test - y_pred)**2)}")

if __name__ == '__main__':
    main()