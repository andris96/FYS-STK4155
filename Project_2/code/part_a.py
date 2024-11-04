#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import autograd.numpy as np
from autograd import grad
from sklearn.preprocessing import StandardScaler


# cost function for OLS
def CostOLS(theta):
    return (1.0 / n) * np.sum((y - X @ theta) ** 2)

# Analytical expression for the gradient
def training_gradient(X, y, theta):
    return -2.0 / len(y) * X.T @ (y - X @ theta)

# Autograd will be relevant later on
#training_gradient = grad(CostOLS)

# Plotting the cost function, to check for oscillations
def plot_cost_history(cost_history):
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost function over iterations")
    plt.show()


# OLS_GD - gradient descent function
def OLS_GD(X, y, Niterations, eta_tune = 0.1, momentum = False, delta_momentum=0.3):
    n = len(y)
    theta = np.random.randn(len(X[0]), 1)  # 3 beta guesses
    H = (2.0 / n) * X.T @ X

    EigValues, EigVectors = np.linalg.eig(H)
    print(f"Eigenvalues of Hessian Matrix:{EigValues}")
    eta = 1 / np.max(EigValues) * eta_tune

    # Using pseudo inverse for (X^T X)^-1 to compare
    theta_linreg = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    print("Own inversion:")
    print(theta_linreg, "\n")

    # Variables for momentum
    change = np.zeros_like(theta) if momentum else None  # Initialize change only if using momentum
    cost_history = []
    for iter in range(Niterations):
        # calculate gradient
        gradients = training_gradient(X, y, theta)
        if momentum:
            # Update with momentum
            new_change = eta * gradients + delta_momentum * change
            theta -= new_change
            change = new_change  # Update change for the next iteration
        else:
            # Standard gradient descent update without momentum
            theta -= eta * gradients

        # Compute the cost at this iteration
        cost = CostOLS(theta)
        cost_history.append(cost)  # Track cost over time

        # stopping criteria
        if np.linalg.norm(gradients) < epsilon:
            print("Converged")
            break

    if momentum:
        plot_cost_history(cost_history)  # Plot cost history to check for oscillations if using momentum

    print("theta from GD with momentum" if momentum else "theta from GD without momentum")
    print(theta, "\n")
    return theta

def ridge_regression_GD(X, y, Niterations, lambda_, eta_tune=[0.1], momentum=False, delta_momentum=0.3):
    n = len(y)
    theta = np.random.randn(len(X[0]), 1)  # Initialize theta with random values
    H = (2.0 / n) * X.T @ X + 2 * lambda_ * np.eye(len(X[0]))
    EigValues, EigVectors = np.linalg.eig(H)

    print(f"Eigenvalues of Hessian Matrix: {EigValues}")

    eta = 1 / np.max(EigValues) * eta_tune

    # Using pseudo inverse for (X^T X)^-1 to compare
    theta_linreg = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    print("Own inversion:")
    print(theta_linreg, "\n")

    # Variables for momentum
    change = np.zeros_like(theta) if momentum else None  # Initialize change only if using momentum
    cost_history = []

    for iteration in range(Niterations):
        # Calculate gradient
        gradient = (2. / n) * X.T @ (X @ theta - y) + 2 * lambda_ * theta

        if momentum:
            # Update with momentum
            new_change = eta * gradient + delta_momentum * change
            theta -= new_change
            change = new_change  # Update change for the next iteration
        else:
            # Standard gradient descent update without momentum
            theta -= eta * gradient

        # Track cost for analysis
        cost = CostOLS(theta)
        cost_history.append(cost)

        # Stopping criterion
        if np.linalg.norm(gradient) < epsilon:
            print("Converged")
            break

    if momentum:
        plot_cost_history(cost_history)  # Plot cost history to check for oscillations if using momentum

    print("theta from GD with momentum" if momentum else "theta from GD without momentum")
    print(theta, "\n")
    return theta


# Time decay learning rate
def learning_schedule(t):
    t0, t1 = 1, 1000
    return t0/(t+t1)

# Stochastic Gradient Descent
# Should tune parameters to get better results
# Could alter the function to take more parameters to better study the effect of these
# Stochastic Gradient Descent
# M is the number of minibatches
# n_epochs is the number of epochs
def SGD(X, y, M=10, n_epochs=50):
    m = int(n / M)  # number of minibatches
    theta = np.zeros((3, 1))  # initialization


    change = np.zeros_like(theta)
    delta_momentum = 0.1


    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = training_gradient(Xi, yi, theta)
            eta = learning_schedule(epoch * m + i)
            # Print to monitor learning rate and gradients
            # print(f"Learning rate at iteration {epoch * m + i}: {eta}")
            # print(f"Gradients at iteration {epoch * m + i}: {gradients}")
            new_change = eta * gradients + delta_momentum * change
            theta -= new_change
            change = new_change
    print("theta from SGD_________________")
    print(theta, "\n")
    return theta


np.random.seed(42)
# Make data set
n = 100
noise = 0.3
x = np.linspace(0, 10, n).reshape(-1, 1) # * noise
# x = np.random.rand(n, 1)
# The true function without noise
# We want to use the analytical expression of the gradients
# to tune the learning rate, so we need to know the true function without noise
y = 4 + 3 * x + 5 * x * x # + noise * np.random.randn(n, 1)

# Design matrix as a function of 2nd order polynomial
X = np.c_[np.ones((n, 1)), x, x * x]

# OLS_GD - setup
eta_tune = [10 ** (-i) for i in range(6)]  # <----- List of scaling factors for the learning rate
lambda_values = [0.001, 0.01, 0.1]
Niterations = 200000
dm = 0.3 # momentum parameter
# stopping criteria
epsilon = 1e-3

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for lambda_ in lambda_values:
    print(f"\nTesting with lambda = {lambda_}")
    ridge_regression_GD(X_scaled, y, Niterations, lambda_, eta_tune=eta_tune[2], momentum=True, delta_momentum=dm)
#%%
