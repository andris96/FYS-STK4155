#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns

"""
Creating a class for FFNN with custom amount of layers and neurons, and different activation functions.

The class will have the following methods:
- create_layers: create layers with Xavier/Glorot initialization
- feed_forward: feed forward through the network and return activations and z values
- backpropagation: backpropagate through the network and return gradients. Different activation functions will be used.
- update_parameters: update weights and biases using gradients
- train_network: train the network using backpropagation and update_parameters
- predict: make predictions

"""

class NeuralNetworkModel:
    def __init__(self, input_size, layer_sizes, activation_funcs):
        """
        Initialize the neural network with specified architecture and activation functions.

        Parameters:
        -----------
        input_size : int
            Number of input features
        layer_sizes : list
            List of integers specifying the size of each layer
        activation_funcs : list
            List of strings specifying the activation function for each layer
            Supported: 'sigmoid', 'relu', 'leaky_relu','linear'
        """
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.activation_funcs = activation_funcs
        self.layers = self.create_layers()
        self.mse_history = []

        # Check if number of layers and activation functions are equal
        if len(layer_sizes) != len(activation_funcs):
            raise ValueError("Number of layers and activation functions must be equal")

        # Validate activation functions
        valid_activations = ['sigmoid', 'relu', 'leaky_relu', 'linear']
        for activation in activation_funcs:
            if activation not in valid_activations:
                raise ValueError(f"Invalid activation function: {activation}")



    # Create layers
    # Returns a list of tuples with weights and biases for each layer
    def create_layers(self):
        layers = []
        input_size = self.input_size

        """
        Different activation functions require different initialization of weights.
        For sigmoid we use Xavier initialization, for ReLU and leaky relu we use He initialization.
        """
        for i, output_size in enumerate(self.layer_sizes):
            # Adjust initialization based on activation function
            if self.activation_funcs[i] in ['relu', 'leaky_relu']:
                # He initialization for (leaky) ReLU
                scale = np.sqrt(2.0 / input_size)
            elif self.activation_funcs[i] == 'sigmoid':
                # Xavier initialization for sigmoid
                scale = np.sqrt(2.0 / (input_size + output_size))
            else:  # linear
                # Xavier initialization for linear
                scale = np.sqrt(1.0 / input_size)

            # Initialize weights based on scaling
            W = np.random.normal(0, scale, (input_size, output_size))
            b = np.zeros(output_size)  # Initialize biases to zero
            layers.append((W, b))
            # Update input size for next layer
            input_size = output_size
        return layers

    # Sigmoid activation function
    def sigmoid(self, z):
        np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # Derivative of sigmoid activation function
    def sigmoid_derivative(self, z):
        zig = self.sigmoid(z)
        return zig * (1 - zig)

    # Linear activation function
    def linear(self, z):
        return z

    # Derivative of linear activation function
    def linear_derivative(self, z):
        return np.ones_like(z)

    # ReLU activation function
    def relu(self, z):
        return np.maximum(0, z)

    # Derivative of ReLU activation function
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    # leaky ReLU activation function
    def leaky_relu(self, z):
        return np.where(z > 0, z, 0.01 * z)

    # Derivative of leaky ReLU activation function
    def leaky_relu_derivative(self, z):
        return np.where(z > 0, 1, 0.01)

    # Mean squared error
    def mse(self, target, predict):
        return np.mean((target - predict) ** 2)

    # R2 score
    def R2(self, target, predict):
        return 1 - np.sum((target - predict) ** 2) / np.sum((target - np.mean(target)) ** 2)

    # Based on the activation function, return the activation function and its derivative
    def get_activations_and_derivatives(self, activation_name):
        activation_map = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'linear': (self.linear, self.linear_derivative),
            'relu': (self.relu, self.relu_derivative),
            'leaky_relu': (self.leaky_relu, self.leaky_relu_derivative)
        }
        return activation_map.get(activation_name)

    # Feed forward
    def feed_forward(self, X):
        activations = [X]
        z_values = []

        for i, (W, b) in enumerate(self.layers):
            # Compute z and activation for each layer
            z = activations[-1] @ W + b
            # Get activation function and its derivative
            activation_func, _ = self.get_activations_and_derivatives(self.activation_funcs[i])
            # Store z and activation
            z_values.append(z)
            activations.append(activation_func(z))

        return activations, z_values

    def backpropagation(self, X, y):
        # Forward pass
        activations, z_values = self.feed_forward(X)
        # Number of samples
        m = X.shape[0]
        # number of layers
        n_layers = len(self.layers)
        # Initialize gradients
        gradients = []
        # Initialize deltas storage
        deltas = [None] * n_layers


        # Output layer error for MSE with any activation function
        _, derivative = self.get_activations_and_derivatives(self.activation_funcs[-1])
        deltas[-1] = (activations[-1] - y) * derivative(z_values[-1])

        # Backpropagate through hidden layers
        for l in reversed(range(n_layers - 1)):
            _, derivative = self.get_activations_and_derivatives(self.activation_funcs[l])
            # Compute delta for each layer
            # delta_l = delta_{l+1} @ W_{l+1}^T * g'(z_l)
            deltas[l] = deltas[l + 1] @ self.layers[l + 1][0].T * derivative(z_values[l])

        # Compute gradients
        for l in range(n_layers):
            # Compute gradients for each layer
            # dW_l = 1/m * a_{l-1}^T @ delta_l
            # db_l = 1/m * sum(delta_l)
            dW = (1 / m) * activations[l].T @ deltas[l]
            db = (1 / m) * np.sum(deltas[l], axis=0)
            gradients.append((dW, db))
        return gradients

    def update_parameters(self, gradients, learning_rate, lambda_reg=0.01):
        # Update weights and biases with L2 regularization
        for l in range(len(self.layers)):
            W, b = self.layers[l]
            dW, db = gradients[l]

            # L2 regularization
            W_reg = W * (1 - learning_rate * lambda_reg)

            # Update parameters with regularization
            W = W_reg - learning_rate * dW
            b = b - learning_rate * db

            self.layers[l] = (W, b)

    def train_network(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate,
            epochs,
            lambda_reg,
            batch_size
    ):
        # Initialize lists to store training and validation loss
        train_losses = []
        val_losses = []

        # Initialize best loss to update during training
        min_loss = np.inf

        # Number of samples
        m = X_train.shape[0]

        # Early stopping parameters
        patience = 5
        patience_counter = 0


        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(m)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Compute gradients
                gradients = self.backpropagation(X_batch, y_batch)

                # Update parameters
                self.update_parameters(gradients, learning_rate, lambda_reg)

            # Compute and store losses every 100 epochs
            if epoch % 100 == 0:
                # Compute and store losses
                train_pred, _ = self.feed_forward(X_train)
                val_pred, _ = self.feed_forward(X_val)

                # Compute loss and store
                train_loss = self.mse(y_train, train_pred[-1])
                val_loss = self.mse(y_val, val_pred[-1])
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Print loss
                """
                print(f"Epoch {epoch}/{epochs},"
                      f"Train MSE: {train_losses[-1]:.4f},"
                      f"Val MSE: {val_losses[-1]:.4f}")
                """
                # Saving best model and implementing early stopping
                if val_loss < min_loss:
                    min_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break


        return train_losses, val_losses


    def predict(self, X):
        """Make predictions"""
        activations, _ = self.feed_forward(X)
        return activations[-1]




    # plot mse as function of epochs
    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
