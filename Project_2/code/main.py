import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from utils.franke import FrankeFunction
import seaborn as sns
from utils.activation_functions import ActivationFunctions
from utils.regression_functions import OLS_GradientDescent, OLS, Ridge, Ridge_GradientDescent, OLS_StochasticGradientDescent, Ridge_StochasticGradientDescent
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.optimizers import find_optimal_eta_OLS_GD, find_optimal_eta_Ridge_GD, find_optimal_eta_batch_OLS_SGD, find_optimal_eta_batch_Ridge_SGD

def generate_data(num_samples, noise=0.1):
    '''
    Generate a simple second order polynomial dataset with added noise
    '''
    np.random.seed(42)
    x = np.linspace(0, 10, num_samples).reshape(-1, 1)
    y = 4 + 3 * x + 5 * x**2 + noise * np.random.randn(num_samples, 1)

    return x, y

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
    franke = FrankeFunction(eps=0.1)
    x_franke, y_franke = franke.generate_data()
    X_franke = franke.create_design_matrix(degree=5)
    franke.plot_franke()

    X_train, X_test, y_train, y_test = train_test_split(X_franke, y_franke, test_size=0.2)

    X_train, X_test = scale_data(X_train, X_test)

    # # Find optimal learning rate for OLS_GD
    # eta_tune = [10**(-i) for i in range(1,6)]
    # best_lr_ols, best_mse, mse_list = find_optimal_eta_OLS_GD(X_train, y_train, X_test, y_test, eta_tune)
    # print(f"OLS_GD best learning rate: {best_lr_ols}")
    # print(f"OLS_GD best MSE: {best_mse}")
    # print(f"OLS_GD MSE list: {np.array(mse_list)}")

    # # Find optimal learning rate and lambda for Ridge_GD
    # lambdas = [10**(-i) for i in range(1,6)]
    # best_lr_ridge, best_lambda_ridge, best_mse, mse_list = find_optimal_eta_Ridge_GD(X_train, y_train, X_test, y_test, eta_tune, lambdas)
    # print(f"Ridge_GD best learning rate: {best_lr_ridge}")
    # print(f"Ridge_GD best lambda: {best_lambda_ridge}")
    # print(f"Ridge_GD best MSE: {best_mse}")
    # print(f"Ridge_GD MSE list: {mse_list}")

    # # Add momentum to OLS_GD and Ridge_GD
    # momentum = 0.5

    # # Find optimal learning rate for OLS_GD with momentum
    # eta_tune = [10**(-i) for i in range(1,6)]
    # best_lr_ols_momentum, best_mse_momentum, mse_list_momentum = find_optimal_eta_OLS_GD(X_train, y_train, X_test, y_test, eta_tune, momentum)
    # print(f"OLS_GD best learning rate: {best_lr_ols_momentum}")
    # print(f"OLS_GD best MSE: {best_mse_momentum}")
    # print(f"OLS_GD MSE list: {np.array(mse_list_momentum)}")

    # # Find optimal learning rate and lambda for Ridge_GD with momentum
    # lambdas = [10**(-i) for i in range(1,6)]
    # best_lr_ridge_momentum, best_lambda_ridge_momentum, best_mse_momentum, mse_list_momentum = find_optimal_eta_Ridge_GD(X_train, y_train, X_test, y_test, eta_tune, lambdas, momentum)
    # print(f"Ridge_GD best learning rate: {best_lr_ridge_momentum}")
    # print(f"Ridge_GD best lambda: {best_lambda_ridge_momentum}")
    # print(f"Ridge_GD best MSE: {best_mse_momentum}")
    # print(f"Ridge_GD MSE list: {mse_list_momentum}")

    # # Find optimal batch size and lr for OLS_SGD
    # batch_sizes = [1, 10, 100, 1000]
    # best_lr_ols_sgd, best_batch_size_ols_sgd, best_mse_ols_sgd, mse_list_ols_sgd, epochs_list_ols_sgd = find_optimal_eta_batch_OLS_SGD(X_train, y_train, X_test, y_test, eta_tune, batch_sizes=batch_sizes)
    # print(f"OLS_SGD best learning rate: {best_lr_ols_sgd}")
    # print(f"OLS_SGD best batch size: {best_batch_size_ols_sgd}")
    # print(f"OLS_SGD best MSE: {best_mse_ols_sgd}")
    # print(f"OLS_SGD MSE list: {mse_list_ols_sgd}")
    # print(f"OLS_SGD epochs list: {epochs_list_ols_sgd}")

    # # Find optimal batch size, lr and lambda for Ridge_SGD
    # lambdas = [10**(-i) for i in range(1,6)]
    # best_lr_ridge_sgd, best_lambda_ridge_sgd, best_batch_size_ridge_sgd, best_mse_ridge_sgd, mse_list_ridge_sgd, epochs_list_ridge_sgd = find_optimal_eta_batch_Ridge_SGD(X_train, y_train, X_test, y_test, eta_tune, lambdas, batch_sizes=batch_sizes)
    # print(f"Ridge_SGD best learning rate: {best_lr_ridge_sgd}")
    # print(f"Ridge_SGD best lambda: {best_lambda_ridge_sgd}")
    # print(f"Ridge_SGD best batch size: {best_batch_size_ridge_sgd}")
    # print(f"Ridge_SGD best MSE: {best_mse_ridge_sgd}")
    # print(f"Ridge_SGD MSE list: {mse_list_ridge_sgd}")
    # print(f"Ridge_SGD epochs list: {epochs_list_ridge_sgd}")

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

    OLS_GD = OLS_GradientDescent(lr=OLS_GD_lr)
    OLS_GD.fit(X_train, y_train)

    OLS_GD_momentum = OLS_GradientDescent(lr=OLS_GD_momentum_lr, momentum=0.5)
    OLS_GD_momentum.fit(X_train, y_train)

    # Ridge_GD = Ridge_GradientDescent(lr=Ridge_GD_lr, lambda_=Ridge_GD_lambda)
    # Ridge_GD.fit(X_train, y_train)

    # Ridge_GD_momentum = Ridge_GradientDescent(lr=Ridge_GD_momentum_lr, lambda_=Ridge_GD_momentum_lambda, momentum=0.5)
    # Ridge_GD_momentum.fit(X_train, y_train)

    OLS_SGD = OLS_StochasticGradientDescent(lr=OLS_SGD_lr, batch_size=OLS_SGD_batch_size)
    OLS_SGD.fit(X_train, y_train)

    # Ridge_SGD = Ridge_StochasticGradientDescent(lr=Ridge_SGD_lr, lambda_=Ridge_SGD_lambda, batch_size=Ridge_SGD_batch_size)
    # Ridge_SGD.fit(X_train, y_train)

    OLS_SGD_momentum = OLS_StochasticGradientDescent(lr=OLS_SGD_lr, batch_size=OLS_SGD_batch_size, momentum=0.5)
    OLS_SGD_momentum.fit(X_train, y_train)

    # Ridge_SGD_momentum = Ridge_StochasticGradientDescent(lr=Ridge_SGD_lr, lambda_=Ridge_SGD_lambda, batch_size=Ridge_SGD_batch_size, momentum=0.5)
    # Ridge_SGD_momentum.fit(X_train, y_train)

    # plot_convergence_rate(OLS_GD, Ridge_GD, OLS_GD_momentum, Ridge_GD_momentum, OLS_SGD, Ridge_SGD, OLS_SGD_momentum, Ridge_SGD_momentum)

    # Add Adagrad to OLS
    OLS_Adagrad = OLS_GradientDescent(lr=OLS_GD_lr, autograd=True, adagrad=True)
    OLS_Adagrad.fit(X_train, y_train)

    OLS_Adagrad_momentum = OLS_GradientDescent(lr=OLS_GD_momentum_lr, momentum=0.5, autograd=True, adagrad=True)
    OLS_Adagrad_momentum.fit(X_train, y_train)

    OLS_SGD_adagrad = OLS_StochasticGradientDescent(lr=OLS_SGD_lr, batch_size=OLS_SGD_batch_size, autograd=True, adagrad=True)
    OLS_SGD_adagrad.fit(X_train, y_train)

    OLS_SGD_adagrad_momentum = OLS_StochasticGradientDescent(lr=OLS_SGD_lr, batch_size=OLS_SGD_batch_size, momentum=0.5, autograd=True, adagrad=True)
    OLS_SGD_adagrad_momentum.fit(X_train, y_train)

    compare_adagrad(OLS_GD, OLS_GD_momentum, OLS_SGD, OLS_SGD_momentum, OLS_Adagrad, OLS_Adagrad_momentum, OLS_SGD_adagrad, OLS_SGD_adagrad_momentum)

if __name__ == '__main__':
    main()