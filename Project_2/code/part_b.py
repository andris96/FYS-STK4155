# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def linear_derivative(z):
    return np.ones_like(z)


def mse(target, predict):
    return np.mean((target - predict) ** 2)

def R2(target, predict):
    return 1 - np.sum((target - predict) ** 2) / np.sum((target - np.mean(target)) ** 2)


def create_design_matrix(x, order):
    X = np.zeros((len(x), order + 1))
    for i in range(order + 1):
        X[:, i] = x[:, 0] ** i
    return X


def create_layers(network_input_size, layer_output_sizes):
    layers = []
    input_size = network_input_size

    # Xavier/Glorot initialization
    for output_size in layer_output_sizes:
        scale = np.sqrt(2.0 / (input_size + output_size))
        W = np.random.normal(0, scale, (input_size, output_size))
        b = np.zeros(output_size)  # Initialize biases to zero
        layers.append((W, b))
        input_size = output_size
    return layers


def feed_forward(X, layers, activation_funcs):
    activations = [X]
    z_values = []

    for (W, b), activation_func in zip(layers, activation_funcs):
        z = activations[-1] @ W + b
        z_values.append(z)
        activations.append(activation_func(z))

    return activations, z_values


def backpropagation(X, y, layers, activation_funcs, activation_ders):
    activations, z_values = feed_forward(X, layers, activation_funcs)
    m = X.shape[0]

    # Initialize gradients
    gradients = []

    # Output layer error
    delta = (activations[-1] - y) * activation_ders[-1](z_values[-1])

    # Backpropagate through layers
    for layer_idx in reversed(range(len(layers))):
        # Compute gradients
        dW = (1 / m) * activations[layer_idx].T @ delta
        db = (1 / m) * np.sum(delta, axis=0)

        gradients.insert(0, (dW, db))

        # Compute delta for next layer
        if layer_idx > 0:
            W = layers[layer_idx][0]
            delta = (delta @ W.T) * activation_ders[layer_idx - 1](
                z_values[layer_idx - 1]
            )

    return gradients


def update_parameters(layers, gradients, learning_rate, lambda_reg=0.01):
    for layer_idx in range(len(layers)):
        W, b = layers[layer_idx]
        dW, db = gradients[layer_idx]

        # L2 regularization
        W_reg = W * (1 - learning_rate * lambda_reg)

        # Update parameters with regularization
        W = W_reg - learning_rate * dW
        b = b - learning_rate * db

        layers[layer_idx] = (W, b)


def train_network(
    X_train,
    y_train,
    X_val,
    y_val,
    layers,
    activation_funcs,
    activation_ders,
    learning_rate,
    lambda_reg,
    n_epochs,
    batch_size,
):
    train_losses = []
    val_losses = []
    m = X_train.shape[0]

    for epoch in range(n_epochs):
        # Shuffle training data
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            gradients = backpropagation(
                X_batch, y_batch, layers, activation_funcs, activation_ders
            )
            update_parameters(layers, gradients, learning_rate, lambda_reg)

        # Compute and store losses
        if epoch % 100 == 0:
            train_pred, _ = feed_forward(X_train, layers, activation_funcs)
            val_pred, _ = feed_forward(X_val, layers, activation_funcs)

            train_loss = mse(y_train, train_pred[-1])
            val_loss = mse(y_val, val_pred[-1])

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Epoch {epoch}/{n_epochs}, Train MSE: {train_loss:.6f}, Val MSE: {val_loss:.6f}"
            )

            # Early stopping check
            if len(val_losses) > 5 and val_losses[-1] > val_losses[-2]:
                patience_counter += 1
                if (
                    patience_counter >= 5
                ):  # Stop if validation loss increases for 5 consecutive checks
                    print("Early stopping triggered")
                    break
            else:
                patience_counter = 0

    return train_losses, val_losses


def find_optimal_lr_lambda(X_train, y_train, X_val, y_val, layers, activation_funcs, activation_ders, learning_rate, lambda_reg, n_epochs, batch_size):
    best_lr = None
    best_lambda = None
    best_mse = np.inf
    best_R2 = -np.inf
    mse_list = []
    R2_list = []

    for lr in learning_rate:
        for lambda_ in lambda_reg:
            layers_copy = [(np.copy(W), np.copy(b)) for W, b in layers]
            train_losses, val_losses = train_network(
                X_train,
                y_train,
                X_val,
                y_val,
                layers_copy,
                activation_funcs,
                activation_ders,
                lr,
                lambda_,
                n_epochs,
                batch_size,
            )

            val_pred, _ = feed_forward(X_val, layers_copy, activation_funcs)
            val_mse = mse(y_val, val_pred[-1])
            val_R2 = R2(y_val, val_pred[-1])
            mse_list.append(val_mse)
            R2_list.append(val_R2)

            if val_mse < best_mse:
                best_mse = val_mse
                best_lr_mse = lr
                best_lambda_mse = lambda_

            if val_R2 > best_R2:
                best_R2 = val_R2
                best_lr_R2 = lr
                best_lambda_R2 = lambda_

    mse_list = np.array(mse_list).reshape(len(learning_rate), len(lambda_reg))
    R2_list = np.array(R2_list).reshape(len(learning_rate), len(lambda_reg))
    return best_lr, best_lambda, best_mse, mse_list, best_R2, R2_list

def plot_heatmap(data, x, y, title):
    sns.heatmap(data, annot=True, fmt=".4f", xticklabels=x, yticklabels=y)
    plt.xlabel("Regularization parameter")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.savefig(f"C:/Users/andre/Documents/FYS-STK4155/Project_2/figures/{title}.png")
    plt.show()


if __name__ == "__main__":

    np.random.seed(42)
    n_samples = 500
    x = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * x + 5 * x * x + np.random.normal(0, 1, (n_samples, 1))

    # Create design matrix and split data
    X = create_design_matrix(x, 2)

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # Network architecture
    network_input_size = X_train.shape[1]
    layer_output_sizes = [8, 8, 1]
    activation_funcs = [sigmoid, sigmoid, lambda x: x]
    activation_ders = [sigmoid_derivative, sigmoid_derivative, linear_derivative]

    # Training parameters
    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    lambda_reg = [0.1, 0.01, 0.001, 0.0001]
    n_epochs = 1000
    batch_size = 32

    # Create and train network
    layers = create_layers(network_input_size, layer_output_sizes)

    best_lr, best_lambda, best_mse, mse_list, best_R2, R2_list = find_optimal_lr_lambda(
        X_train,
        y_train,
        X_val,
        y_val,
        layers,
        activation_funcs,
        activation_ders,
        learning_rate,
        lambda_reg,
        n_epochs,
        batch_size,
    )

    # Plot heatmap of MSE values
    plot_heatmap(mse_list, lambda_reg, learning_rate, "Validation MSE")
    plot_heatmap(R2_list, lambda_reg, learning_rate, "Validation R2")

    np.random.seed(42)
    n_samples = 500
    x = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * x + 5 * x * x + np.random.normal(0, 1, (n_samples, 1))

    # Create design matrix and split data
    X = create_design_matrix(x, 2)

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # Network architecture
    network_input_size = X_train.shape[1]
    layer_output_sizes = [8, 8, 1]
    activation_funcs = [sigmoid, sigmoid, lambda x: x]
    activation_ders = [sigmoid_derivative, sigmoid_derivative, linear_derivative]

#%%
from sklearn.neural_network import MLPRegressor

# Create and train network
DNN_scikit = np.zeros((len(learning_rate), len(lambda_reg)), dtype=object)

for i, lr in enumerate(learning_rate):
    for j, lambda_ in enumerate(lambda_reg):
        dnn = MLPRegressor(hidden_layer_sizes=(8, 8), activation="logistic", solver="adam", alpha=lambda_, learning_rate_init=lr, max_iter=n_epochs, batch_size=batch_size, random_state=42)
        dnn.fit(X_train, y_train.ravel())
        DNN_scikit[i, j] = dnn

# Plot heatmap of MSE values with scikit-learn
mse_list_scikit = np.zeros((len(learning_rate), len(lambda_reg)))
R2_list_scikit = np.zeros((len(learning_rate), len(lambda_reg)))

for i in range(len(learning_rate)):
    for j in range(len(lambda_reg)):
        dnn = DNN_scikit[i, j]
        y_pred = dnn.predict(X_val)
        mse_list_scikit[i, j] = mean_squared_error(y_val, y_pred)
        R2_list_scikit[i, j] = dnn.score(X_val, y_val)

plot_heatmap(mse_list_scikit, lambda_reg, learning_rate, "Validation MSE (scikit-learn)")


#%%
