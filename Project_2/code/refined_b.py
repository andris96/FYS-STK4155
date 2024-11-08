#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import seaborn as sns
from utils.FFNN import NeuralNetworkModel
#from utils.franke import franke_function


# Create design matrix
# x: array, input data
# order: int, polynomial order
def create_design_matrix(x, order):
    X = np.zeros((len(x), order + 1))
    for i in range(order + 1):
        X[:, i] = x[:, 0] ** i
    return X

def franke_function(x, y):
    """
    The 2D Franke function.
    """
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4

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
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

# Create and train the neural network model
model = NeuralNetworkModel(input_size=X_train.shape[1], layer_sizes=[8, 4, 1], activation_funcs=['relu', 'relu', 'linear'])
train_losses, val_losses = model.train_network(
    X_train, y_train, X_val, y_val,
    learning_rate=0.01, epochs=1000, lambda_reg=0.01, batch_size=32
)

learning_rates = [0.1, 0.01, 0.001, 0.0001]
lambdas = [0.1, 0.01, 0.001, 0.0001]

# Plot heatmap of learning rates and lambda values
model.plot_heatmap_lr_lambda(X_train, y_train, X_val, y_val, learning_rates, lambdas)




#%%
