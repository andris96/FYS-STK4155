#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
from utils.FFNN import NeuralNetworkModel



def franke_function(x, y, sample_size=500):
    """
    The 2D Franke function.
    """
    noise = 0.1 * np.random.randn(sample_size,1)
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4 + noise

def plot_heatmap_lr_lambda(X_train, y_train, X_val, y_val, learning_rates, lambdas, layer_sizes, activation_funcs):
    """
    Plot a heatmap of validation loss for different learning rates and lambda values.

    Parameters:
    -----------
    X_train: numpy.ndarray
        Training input data
    y_train: numpy.ndarray
        Training target data
    X_val: numpy.ndarray
        Validation input data
    y_val: numpy.ndarray
        Validation target data
    learning_rates: list
        List of learning rate values to test
    lambdas: list
        List of lambda (regularization) values to test
    """
    val_losses = []

    for lr in learning_rates:
        row_losses = []
        for lam in lambdas:
            model = NeuralNetworkModel(input_size=X_train.shape[1], layer_sizes=layer_sizes, activation_funcs=activation_funcs)
            _, val_loss = model.train_network(X_train,
                                                y_train,
                                                X_val,
                                                y_val,
                                                learning_rate=lr,
                                                epochs=1000,
                                                lambda_reg=lam,
                                                batch_size=32)
            row_losses.append(val_loss[-1])
        val_losses.append(row_losses)

    plt.figure(figsize=(10, 8))
    sns.heatmap(val_losses, xticklabels=lambdas, yticklabels=learning_rates, cmap="YlOrRd")
    plt.title("Validation Loss Heatmap", fontsize=18)
    plt.xlabel("Lambda", fontsize=18)
    plt.ylabel("Learning Rate", fontsize=18)
    plt.savefig("../images/heatmap_lr_lambda_FFNN.png")
    plt.show()

# Generate data
n_samples = 500
x = np.random.rand(n_samples, 1)
y = np.random.rand(n_samples, 1)
z = franke_function(x, y)

# Create design matrix and split data
X = np.hstack((x, y))

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, z, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)
y_test = scaler.transform(y_test)

learning_rates = [0.1, 0.01, 0.001, 0.0001]
lambdas = [0.1, 0.01, 0.001, 0.0001]
epochs = 2000
layer_sizes = [50, 50, 1]
activation_funcs = ['sigmoid', 'sigmoid', 'linear']

# Create and train the neural network model
model = NeuralNetworkModel(input_size=X_train.shape[1],
                           layer_sizes= layer_sizes,
                           activation_funcs=activation_funcs)
train_losses, val_losses = model.train_network(
    X_train, y_train, X_val, y_val,
    learning_rate=0.01, epochs=epochs, lambda_reg=0.01, batch_size=32
)


# Plot heatmap of learning rates and lambda values
plot_heatmap_lr_lambda(X_train,
                       y_train,
                       X_val,
                       y_val,
                       learning_rates,
                       lambdas,layer_sizes,
                       activation_funcs)


# By evaluating the heatloss plot, we can see that the
# optimal learning rate is 0.1 and the optimal lambda is 0.0001

learning_rate_best = 0.1
lambda_best = 0.0001

model_best_sigmoid = NeuralNetworkModel(
    input_size=X_train.shape[1],
    layer_sizes=layer_sizes,
    activation_funcs=activation_funcs
)
train_losses, val_losses = model_best_sigmoid.train_network(
    X_train, y_train, X_val, y_val,
    learning_rate=learning_rate_best, epochs=epochs, lambda_reg=lambda_best, batch_size=32
)

plt.plot(val_losses, label="Validation Loss")
plt.plot(train_losses, label="Training Loss")
plt.legend()
plt.xlabel("Epoch * 100")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.savefig("../images/training_validation_loss_FFNN.png")
plt.show()

from sklearn.metrics import mean_squared_error
y_pred = model_best_sigmoid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error with sigmoid: {mse}")

model_relu = NeuralNetworkModel(
    input_size=X_train.shape[1],
    layer_sizes=layer_sizes,
    activation_funcs=['relu', 'relu', 'linear']
)
train_losses, val_losses = model_relu.train_network(
    X_train, y_train, X_val, y_val,
    learning_rate=learning_rate_best, epochs=epochs, lambda_reg=lambda_best, batch_size=32
)

y_pred = model_relu.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error with relu: {mse}")

model_relu_leaky = NeuralNetworkModel(
    input_size=X_train.shape[1],
    layer_sizes=layer_sizes,
    activation_funcs=['leaky_relu', 'leaky_relu', 'linear']
)
train_losses, val_losses = model_relu_leaky.train_network(
    X_train, y_train, X_val, y_val,
    learning_rate=learning_rate_best, epochs=epochs, lambda_reg=lambda_best, batch_size=32
)


y_pred = model_relu_leaky.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error with leaky relu: {mse}")

from sklearn.datasets import load_breast_cancer

# Load the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Reshape target variables to match network output
y_train = y_train.reshape(-1, 1)  # Reshape to (n_samples, 1)
y_test = y_test.reshape(-1, 1)    # Reshape to (n_samples, 1)

# Create and train the neural network model
model_cancer = NeuralNetworkModel(
    input_size=X_train_scaled.shape[1],
    layer_sizes=[50, 50, 1],
    activation_funcs=['sigmoid', 'sigmoid', 'sigmoid']
)
train_losses, val_losses = model_cancer.train_network(
    X_train_scaled, y_train, X_test_scaled, y_test,
    learning_rate=learning_rate_best, epochs=epochs, lambda_reg=lambda_best, batch_size=32
)

y_pred = model_cancer.predict(X_test_scaled)

y_pred = np.where(y_pred > 0.5, 1, 0)

# Accuracy score
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

# Confusion matrix with seaborn
from sklearn.metrics import confusion_matrix



conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted", fontsize=18)
plt.ylabel("Actual", fontsize=18)
plt.title("Confusion Matrix", fontsize=18)
plt.savefig("../images/confusion_matrix_cancer_data.png")
plt.show()





#%%






#%%
