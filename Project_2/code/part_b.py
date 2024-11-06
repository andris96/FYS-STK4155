# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import autograd.numpy as np
from autograd import grad
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse(target, predict):
    return 1/2 * sum((target-predict)**2)

def mse_der(target, predict):
    return (predict-target)

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        # Initialize weights with a normal distribution (mean=0, std deviation=0.01)
        W = np.random.randn(i_size, layer_output_size) * 0.01
        # Initialize biases to zeros
        b = np.zeros(layer_output_size)
        layers.append((W, b))
        i_size = layer_output_size
    return layers



def feed_forward_batch(input, layers, activation_funcs):
    a = input
    for i, ((W, b), activation_func) in enumerate(zip(layers, activation_funcs)):
        z = np.matmul(a, W) + b
        if i == len(layers) - 1:
            a = z  # Linear activation for the output layer
        else:
            a = activation_func(z)
            # Print activations to monitor saturation
            print(f"Layer {i} activation range: min={a.min()}, max={a.max()}")
    return a


def cost_batch(layers, input, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target)

def feed_forward_saver(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input

    for (W, b), activation_func in zip(layers, activation_funcs):
        # print("Shape of a before multiplication:", a.shape)
        # print("Shape of W:", W.shape)
        # print("Shape of b:", b.shape)
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

def backpropagation_batch(
    input, layers, activation_funcs, target, activation_ders, cost_der=mse_der
):
    # Forward pass to get intermediate values
    layer_inputs, zs, predict = feed_forward_saver(input, layers, activation_funcs)

    # Initialize a list to store gradients for each layer
    layer_grads = [() for layer in layers]

    # Calculate delta for the output layer
    delta = cost_der(predict, target) * activation_ders[-1](zs[-1])  # δ^L = ∇_a C ⊙ σ'(z^L)

    # Loop through the layers in reverse (backward pass)
    for i in reversed(range(len(layers))):
        # Calculate ∂C/∂b^l_j = δ^l_j
        dC_db = np.sum(delta, axis=0)  # Sum over the batch to get shape (output size,)

        # Calculate ∂C/∂w^l_{jk} = a^{l-1}_k δ^l_j
        layer_input = layer_inputs[i]  # a^{l-1}
        dC_dW = layer_input.T @ delta  # Shape: (input size, output size)

        # Store the gradients
        layer_grads[i] = (dC_dW, dC_db)

        # Calculate delta for the previous layer, if not at the input layer
        if i > 0:
            W, _ = layers[i]
            delta = (delta @ W.T) * activation_ders[i - 1](zs[i - 1])  # δ^l = ((w^{l+1})^T δ^{l+1}) ⊙ σ'(z^l)

    return layer_grads

def update_weights_batch(layers, layer_grads, learning_rate):
    for i in range(len(layers)):
        W, b = layers[i]
        dC_dW, dC_db = layer_grads[i]

        W -= learning_rate * dC_dW
        b -= learning_rate * dC_db

def train_batch(
    input, target, layers, activation_funcs, activation_ders, learning_rate, Niterations
):
    for i in range(Niterations):
        layer_grads = backpropagation_batch(input, layers, activation_funcs, target, activation_ders)
        update_weights_batch(layers, layer_grads, learning_rate)



n = 100
x = np.linspace(0, 10, n).reshape(-1, 1)
y = 4 + 3 * x + 5 * x * x
X = np.c_[np.ones((n, 1)), x, x * x]

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=train_size,
                                                    test_size=test_size)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

network_input_size = X_train_scaled.shape[1]
layer_output_sizes = [300, 1]
activation_funcs = [sigmoid, sigmoid]
activation_ders = [sigmoid_derivative, sigmoid_derivative]
learning_rate = 1e-4

layers = create_layers_batch(network_input_size, layer_output_sizes)

train_batch(X_train_scaled, Y_train, layers, activation_funcs, activation_ders, learning_rate, 10000)

Y_predict = feed_forward_batch(X_test_scaled, layers, activation_funcs)



print(f"MSE: {mse(Y_test, Y_predict)}")


plt.scatter(Y_test, Y_predict, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')  # Perfect prediction line
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs. Predicted Values")
plt.show()
#%%