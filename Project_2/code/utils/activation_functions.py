import numpy as np

class ActivationFunctions():
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(x):
        return np.maximum(0, x)

    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha*x)

    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
