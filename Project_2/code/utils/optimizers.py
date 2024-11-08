import numpy as np
from tqdm import tqdm
from utils.regression_functions import OLS_GradientDescent, Ridge_GradientDescent, OLS_StochasticGradientDescent, Ridge_StochasticGradientDescent

def find_optimal_eta_OLS_GD(X_train, y_train, X_test, y_test, eta_tune, momentum=None):
    best_lr = None
    best_mse = np.inf
    mse_list = []

    for lr in tqdm(eta_tune):
        OLS_GD = OLS_GradientDescent(lr=lr, momentum=momentum)
        OLS_GD.fit(X_train, y_train)
        OLS_pred = OLS_GD.predict(X_test)
        OLS_GD_mse = np.mean((y_test - OLS_pred)**2)
        mse_list.append(OLS_GD_mse)

        if OLS_GD_mse < best_mse:
            best_mse = OLS_GD_mse
            best_lr = lr

    return best_lr, best_mse, mse_list

def find_optimal_eta_Ridge_GD(X_train, y_train, X_test, y_test, eta_tune, lambdas, momentum=None):
    best_lr = None
    best_mse = np.inf
    best_lambda = None
    mse_list = np.zeros((len(eta_tune), len(lambdas)))

    for lr in tqdm(eta_tune):
        for lambda_ in lambdas:
            Ridge_GD = Ridge_GradientDescent(lr=lr, lambda_=lambda_, momentum=momentum)
            Ridge_GD.fit(X_train, y_train)
            Ridge_pred = Ridge_GD.predict(X_test)
            Ridge_GD_mse = np.mean((y_test - Ridge_pred)**2)
            mse_list[eta_tune.index(lr), lambdas.index(lambda_)] = Ridge_GD_mse

            if Ridge_GD_mse < best_mse:
                best_mse = Ridge_GD_mse
                best_lr = lr
                best_lambda = lambda_

    return best_lr, best_lambda, best_mse, mse_list

def find_optimal_eta_batch_OLS_SGD(X_train, y_train, X_test, y_test, eta_tune, momentum=None, batch_sizes=1, decay=0.1, epochs=1000, early_stopping_threshold=1e-6, patience=100, lr_patience=50):
    best_lr = None
    best_batch_size = None
    best_mse = np.inf
    mse_list = np.zeros((len(eta_tune), len(batch_sizes)))
    epochs_list = np.zeros((len(eta_tune), len(batch_sizes)))

    for batch_size in tqdm(batch_sizes):
        for lr in eta_tune:
            OLS_SGD = OLS_StochasticGradientDescent(lr=lr, momentum=momentum, batch_size=batch_size, decay=decay, epochs=epochs, early_stopping_threshold=early_stopping_threshold, patience=patience, lr_patience=lr_patience)
            OLS_SGD.fit(X_train, y_train)
            OLS_pred = OLS_SGD.predict(X_test)
            OLS_SGD_mse = np.mean((y_test - OLS_pred)**2)
            mse_list[eta_tune.index(lr), batch_sizes.index(batch_size)] = OLS_SGD_mse
            epochs_list[eta_tune.index(lr), batch_sizes.index(batch_size)] = OLS_SGD.epochs_ran

            if OLS_SGD_mse < best_mse:
                best_mse = OLS_SGD_mse
                best_lr = lr
                best_batch_size = batch_size

    return best_lr, best_batch_size, best_mse, mse_list, epochs_list

def find_optimal_eta_batch_Ridge_SGD(X_train, y_train, X_test, y_test, eta_tune, lambdas, momentum=None, batch_sizes=1, decay=1):
    best_lr = None
    best_mse = np.inf
    best_batch_size = None
    best_lambda = None
    mse_list = np.zeros((len(eta_tune), len(lambdas)))
    epochs_list = np.zeros((len(eta_tune), len(lambdas)))

    for batch_size in tqdm(batch_sizes):
        for lr in tqdm(eta_tune):
            for lambda_ in lambdas:
                Ridge_SGD = Ridge_StochasticGradientDescent(lr=lr, lambda_=lambda_, momentum=momentum, batch_size=batch_size, decay=decay)
                Ridge_SGD.fit(X_train, y_train)
                Ridge_pred = Ridge_SGD.predict(X_test)
                Ridge_SGD_mse = np.mean((y_test - Ridge_pred)**2)
                mse_list[eta_tune.index(lr), lambdas.index(lambda_)] = Ridge_SGD_mse
                epochs_list[eta_tune.index(lr), lambdas.index(lambda_)] = Ridge_SGD.epochs_ran

                if Ridge_SGD_mse < best_mse:
                    best_mse = Ridge_SGD_mse
                    best_lr = lr
                    best_lambda = lambda_
                    best_batch_size = batch_size


    return best_lr, best_lambda, best_batch_size, best_mse, mse_list, epochs_list
