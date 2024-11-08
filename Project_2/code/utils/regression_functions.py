import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autograd import grad
from autograd import numpy as anp

class OLS():
    def __init__(self):
        self.random_state = 42

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.beta = np.random.randn(X.shape[1], 1)
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ y

    def predict(self, X):
        return X @ self.beta

class OLS_GradientDescent():
    def __init__(self, lr=0.01, epochs=1000, momentum=None, adagrad=False, autograd=False):
        self.lr = lr
        self.epochs = epochs
        self.random_state = 42
        self.momentum = momentum
        self.autograd = autograd
        self.adagrad = adagrad
        self.mse_history = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.y = y.reshape(-1, 1)
        self.beta = np.random.randn(X.shape[1], 1)
        self.v = np.zeros_like(self.beta)
        self.G = np.zeros_like(self.beta)
        n = len(y)

        if self.autograd:
            training_grad = grad(self.CostOLS, argnum=2)

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            if self.autograd:
                gradient = training_grad(self.y, self.X, self.beta)
            else:
                gradient = (2.0/n) * self.X.T @ (self.X @ self.beta - self.y)
            if self.adagrad:
                self.G += gradient**2
                self.beta -= self.lr * gradient / np.sqrt(self.G + 1e-8)
                if self.momentum:
                    self.v = self.momentum * self.v + self.lr * gradient
                    self.beta -= self.v
            elif self.momentum:
                self.v = self.momentum * self.v + self.lr * gradient
                self.beta -= self.v
            else:
                self.beta -= self.lr * gradient

            self.mse_history.append(np.mean((self.y - self.X @ self.beta)**2))

        self.beta = self.beta.flatten()

    def CostOLS(self, y, X, beta):
        return (1.0/len(y)) * anp.sum((y - X @ beta)**2)

    def predict(self, X):
        return X @ self.beta
    
class OLS_StochasticGradientDescent():
    def __init__(self, lr=0.01, epochs=1000, momentum=None, batch_size=1, decay=0.1, early_stopping_threshold=1e-6, patience=100, lr_patience=25, adagrad=False, autograd=False):
        self.lr = lr
        self.epochs = epochs
        self.random_state = 42
        self.momentum = momentum
        self.batch_size = batch_size
        self.num_batches = None
        self.autograd = autograd
        self.adagrad = adagrad
        self.mse_history = []
        self.epochs_ran = 0
        self.decay = decay
        self.early_stopping_threshold = early_stopping_threshold
        self.patience = patience
        self.lr_patience = lr_patience

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.y = y.reshape(-1, 1)
        self.beta = np.random.randn(X.shape[1], 1)
        self.v = np.zeros_like(self.beta)
        self.G = np.zeros_like(self.beta)
        n = len(y)   
        self.num_batches = n // self.batch_size

        patience_counter = 0
        lr_patience_counter = 0
        best_mse = np.inf

        if self.autograd:
            training_grad = grad(self.CostOLS, argnum=2)

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            if self.num_batches == 1:
                # Full batch gradient descent
                gradient = (2.0/n) * self.X.T @ (self.X @ self.beta - self.y)
                if self.adagrad:
                    self.G += gradient**2
                    self.beta -= self.lr * gradient / np.sqrt(self.G + 1e-8)
                    if self.momentum:
                        self.v = self.momentum * self.v + self.lr * gradient
                        self.beta -= self.v
                elif self.momentum:
                    self.v = self.momentum * self.v + self.lr * gradient
                    self.beta -= self.v
                else:
                    self.beta -= self.lr * gradient
            else:
                # Stochastic gradient descent with mini-batches
                for i in range(self.num_batches):
                    random_index = self.batch_size * np.random.randint(self.num_batches)
                    X_batch = self.X[random_index : random_index + self.batch_size, :]
                    y_batch = self.y[random_index : random_index + self.batch_size, :]

                    if self.autograd:
                        gradient = (1.0/self.batch_size)*training_grad(y_batch, X_batch, self.beta)
                    else:
                        gradient = (2.0/self.batch_size) * X_batch.T @ (X_batch @ self.beta - y_batch)
                    
                    if self.adagrad:
                        self.G += gradient**2
                        self.beta -= self.lr * gradient / np.sqrt(self.G + 1e-8)
                        if self.momentum:
                            self.v = self.momentum * self.v + self.lr * gradient
                            self.beta -= self.v
                    elif self.momentum:
                        self.v = self.momentum * self.v + self.lr * gradient
                        self.beta -= self.v
                    else:
                        self.beta -= self.lr * gradient

            mse = np.mean((y - self.X @ self.beta)**2)
            self.mse_history.append(mse)
            
            if mse < best_mse - self.early_stopping_threshold:
                best_mse = mse
                patience_counter = 0
                lr_patience_counter = 0
            else:
                patience_counter += 1
                lr_patience_counter += 1

            if patience_counter == self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.epochs_ran = epoch
                break

            if lr_patience_counter == self.lr_patience:
                self.lr *= self.decay
                print(f"Reducing learning rate to {self.lr}")
                lr_patience_counter = 0
            
        self.beta = self.beta.flatten()
        self.epochs_ran = epoch

    def CostOLS(self, y, X, beta):
        return anp.sum((y - X @ beta)**2)

    def predict(self, X):
        return X @ self.beta
    
class Ridge():
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda
        self.random_state = 42

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.beta = np.random.randn(X.shape[1], 1)
        self.beta = np.linalg.inv(self.X.T @ self.X + self.lmbda * np.eye(X.shape[1])) @ self.X.T @ y

    def predict(self, X):
        return X @ self.beta

class Ridge_GradientDescent():
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.1, momentum=False):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.random_state = 42
        self.momentum = momentum

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.y = y.reshape(-1, 1)
        self.beta = np.random.randn(X.shape[1], 1)
        self.v = np.zeros_like(self.beta)
        self.mse_history = []
        n = len(y)

        for _ in tqdm(range(self.epochs), desc="Epochs"):
            
            gradient = (2.0/n) * self.X.T @ (self.X @ self.beta - self.y) + 2 * self.lambda_ * self.beta
            
            if self.momentum:
                self.v = self.momentum * self.v + self.lr * gradient
                self.beta -= self.v
            else:
                self.beta -= self.lr * gradient

            self.mse_history.append(np.mean((self.y - self.X @ self.beta)**2))

        self.beta = self.beta.flatten()

    def predict(self, X):
        return X @ self.beta
    
class Ridge_StochasticGradientDescent():
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.1, momentum=False, batch_size=1, decay=0.1, early_stopping_threshold=1e-6, patience=10, lr_patience=5):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.random_state = 42
        self.batch_size = batch_size
        self.momentum = momentum
        self.decay = decay
        self.early_stopping_threshold = early_stopping_threshold
        self.patience = patience
        self.lr_patience = lr_patience
        self.mse_history = []
        self.epochs_ran = 0

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.X = X
        self.y = y.reshape(-1, 1)
        self.beta = np.random.randn(X.shape[1], 1)
        self.N_batches = int(self.X.shape[0] / self.batch_size)
        self.v = np.zeros_like(self.beta)
        n = len(y)

        patience_counter = 0
        lr_patience_counter = 0
        best_mse = np.inf

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            random_index = np.random.permutation(n)
            X_shuffled = self.X[random_index]
            y_shuffled = self.y[random_index]

            for i in range(0, n, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                gradient = (2.0/self.X.shape[0]) * X_batch.T @ (X_batch @ self.beta - y_batch) + 2 * self.lambda_ * self.beta
                if self.momentum:
                    self.v = self.momentum * self.v + self.lr * gradient
                    self.beta -= self.v
                else:
                    self.beta -= self.lr * gradient

            mse = np.mean((y - self.X @ self.beta)**2)
            self.mse_history.append(mse)

            if mse < best_mse - self.early_stopping_threshold:
                best_mse = mse
                patience_counter = 0
                lr_patience_counter = 0
            else:
                patience_counter += 1
                lr_patience_counter += 1

            if patience_counter == self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.epochs_ran = epoch
                break

            if lr_patience_counter == self.lr_patience:
                self.lr *= self.decay
                print(f"Reducing learning rate to {self.lr}")
                lr_patience_counter = 0

        self.beta = self.beta.flatten()
        self.epochs_ran = epoch

    def predict(self, X):
        return X @ self.beta