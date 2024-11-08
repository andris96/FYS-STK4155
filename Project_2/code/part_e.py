import numpy as np
import tqdm as tqdm_func
import seaborn as sns
import matplotlib.pyplot as plt

# Helper funcitons------------------------------------------------------------


def add_bias(X, bias):
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias  # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis=1)


def accuracy(predicted, gold):
    return np.mean(predicted == gold)


class Ridge_StochasticGradientDescent:
    def __init__(
        self,
        lr=0.1,
        epochs=1000,
        lambda_=0.001,
        momentum=0.7,
        batch_size=100,
        decay=0.1,
        early_stopping_threshold=1e-6,
        patience=10,
        lr_patience=5,
    ):
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

        for epoch in tqdm_func(range(self.epochs), desc="Epochs"):
            random_index = np.random.permutation(n)
            X_shuffled = self.X[random_index]
            y_shuffled = self.y[random_index]

            for i in range(0, n, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]
                gradient = (2.0 / self.X.shape[0]) * X_batch.T @ (
                    X_batch @ self.beta - y_batch
                ) + 2 * self.lambda_ * self.beta
                if self.momentum:
                    self.v = self.momentum * self.v + self.lr * gradient
                    self.beta -= self.v
                else:
                    self.beta -= self.lr * gradient

            mse = np.mean((y - self.X @ self.beta) ** 2)
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


def logistic(X):
    return 1 / (1 + np.exp(-X))


def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)


"Part a)"


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Compute binary cross entropy loss
    Adding epsilon to avoid log(0)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_gradient(X, y_true, y_pred):
    """
    Compute gradient of binary cross entropy loss
    """
    return X.T @ (y_pred - y_true)


class LogRegClass(Ridge_StochasticGradientDescent):
    def __init__(
        self,
        bias=-1,
        lr=0.1,
        epochs=1000,
        lambda_=0.001,
        momentum=0.7,
        batch_size=100,
        decay=0.1,
        early_stopping_threshold=1e-6,
        patience=10,
        lr_patience=5,
    ):
        super().__init__(
            lr=lr,
            epochs=epochs,
            lambda_=lambda_,
            momentum=momentum,
            batch_size=batch_size,
            decay=decay,
            early_stopping_threshold=early_stopping_threshold,
            patience=patience,
            lr_patience=lr_patience,
        )
        self.bias = bias

    def fit(self, X_train, t_train, val_set=None):
        # Add bias term to input features
        X_train = add_bias(X_train, self.bias)
        (N, m) = X_train.shape

        # Initialize weights and momentum
        self.weights = np.zeros(m)
        self.v = np.zeros_like(self.weights)

        threshold = 0.5

        # Initialize metrics storage
        self.list_acc = []
        self.list_bce = []  # Changed from MSE to BCE
        self.list_val_acc = []
        self.list_val_bce = []  # Changed from MSE to BCE
        self.no_epochs_training = []
        self.n_epochs_trained = 0

        patience_counter = 0
        lr_patience_counter = 0
        best_bce = np.inf

        for epoch in range(self.epochs):
            self.no_epochs_training.append(epoch)

            # Create mini-batches using random permutation
            random_index = np.random.permutation(N)
            X_shuffled = X_train[random_index]
            t_shuffled = t_train[random_index]

            for i in range(0, N, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                t_batch = t_shuffled[i : i + self.batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute gradient with binary cross entropy and L2 regularization
                gradient = (1.0 / self.batch_size) * binary_cross_entropy_gradient(
                    X_batch, t_batch, y_pred
                ) + 2 * self.lambda_ * self.weights

                # Update weights using momentum
                if self.momentum:
                    self.v = self.momentum * self.v + self.lr * gradient
                    self.weights -= self.v
                else:
                    self.weights -= self.lr * gradient

            # Compute training metrics
            y_pred_full = self.forward(X_train)
            current_bce = binary_cross_entropy(t_train, y_pred_full)
            self.list_bce.append(current_bce)
            self.list_acc.append(
                accuracy((y_pred_full > threshold).astype("int"), t_train)
            )

            # Compute validation metrics if validation set is provided
            if val_set is not None:
                X_val, t_val = val_set
                X_val_bias = add_bias(X_val, self.bias)
                y_val_pred = self.forward(X_val_bias)
                self.list_val_bce.append(binary_cross_entropy(t_val, y_val_pred))
                self.list_val_acc.append(
                    accuracy((y_val_pred > threshold).astype("int"), t_val)
                )

            # Early stopping logic based on BCE
            if self.list_val_bce[-1] < best_bce - self.early_stopping_threshold:
                best_bce = self.list_val_bce[-1]
                patience_counter = 0
                lr_patience_counter = 0
            else:
                patience_counter += 1
                lr_patience_counter += 1

            # Check if we should stop training
            if patience_counter == self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Check if we should reduce learning rate
            if lr_patience_counter == self.lr_patience:
                self.lr *= self.decay
                print(f"Reducing learning rate to {self.lr}")
                lr_patience_counter = 0

        self.n_epochs_trained = epoch + 1

    def forward(self, X):
        return logistic(X @ self.weights)

    def predict(self, X, threshold=0.5):
        z = add_bias(X, self.bias)
        return (self.forward(z) > threshold).astype("int")

    def predict_probability(self, X):
        z = add_bias(X, self.bias)
        return self.forward(z)

    def plot_confusion_matrix(self, X_test, y_test, figsize=(10, 8)):
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create figure
        plt.figure(figsize=figsize)

        # Create heatmap with raw counts
        sns.heatmap(
            cm,
            annot=False,  # We'll add custom annotations
            cmap="Blues",
            square=True,
            linewidths=0.5,
            linecolor="gray",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )

        # Add count and percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if cm_percent[i, j] > 50 else "black",
                )

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Add metrics text
        metrics_text = (
            f"Accuracy: {accuracy:.1f}%\n"
            f"Precision: {precision:.1f}%\n"
            f"Recall: {recall:.1f}%\n"
            f"F1 Score: {f1:.1f}%"
        )
        plt.figtext(1.15, 0.5, metrics_text, fontsize=12, va="center")

        plt.title("Confusion Matrix", pad=20, fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")

        # Add meaning of the matrix cells
        cell_meanings = (
            "TN: True Negative\nFN: False Negative\n"
            "FP: False Positive\nTP: True Positive"
        )
        plt.figtext(1.15, 0.2, cell_meanings, fontsize=10, style="italic")

        plt.tight_layout()
        plt.show()

        # Print detailed classification report
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(y_test, y_pred))


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load and prepare data
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=42, test_size=0.2
    )

    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, random_state=42, test_size=0.2
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogRegClass(
        bias=-1,
        lr=0.01,
        epochs=1000,
        lambda_=0.001,
        momentum=0.7,
        batch_size=32,
        decay=0.1,
        patience=10,
    )


def plot_lambda_comparison(X_train, t_train, val_set=None):
    # Fixed hyperparameters
    fixed_params = {
        "lr": 0.01,
        "epochs": 1000,
        "momentum": 0.9,
        "decay": 0.1,
        "early_stopping_threshold": 1e-6,
        "patience": 10,
        "lr_patience": 5,
    }

    # Lambda values to test
    lambda_values = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [16, 128]

    # Enhanced color scheme
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]  # Distinct, vibrant colors
    alpha_train = 0.9  # Increased visibility for training lines
    alpha_val = 0.7  # Validation lines more transparent

    # Set the style using seaborn
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Create separate plots for each batch size and metric
    for batch_size in batch_sizes:
        models = []  # Store models to avoid retraining

        # Train models for this batch size
        for lambda_, color in zip(lambda_values, colors):
            model = LogRegClass(batch_size=batch_size, lambda_=lambda_, **fixed_params)
            model.fit(X_train, t_train, val_set)
            models.append(model)

        # Plot 1: Accuracy
        fig, ax = plt.subplots(figsize=(12, 8))

        min_acc = float("inf")
        max_acc = float("-inf")

        for model, lambda_, color in zip(models, lambda_values, colors):
            epochs_trained = len(model.list_acc)
            epochs_range = range(1, epochs_trained + 1)

            min_acc = min(min_acc, min(model.list_acc))
            max_acc = max(max_acc, max(model.list_acc))

            ax.plot(
                epochs_range,
                model.list_acc,
                color=color,
                label=f"λ={lambda_} (Train)",
                alpha=alpha_train,
                linewidth=2.0,
            )

            if val_set is not None and model.list_val_acc:
                min_acc = min(min_acc, min(model.list_val_acc))
                max_acc = max(max_acc, max(model.list_val_acc))
                ax.plot(
                    epochs_range,
                    model.list_val_acc,
                    color=color,
                    linestyle="--",
                    label=f"λ={lambda_} (Val)",
                    alpha=alpha_val,
                    linewidth=1.6,
                )

        # Enhanced y-axis limits and grid
        y_padding = (max_acc - min_acc) * 0.1
        ax.set_ylim(max(0, min_acc - y_padding), min(1, max_acc + y_padding))
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)  # Place grid behind data

        ax.set_title(
            f"Model Accuracy, MB = {batch_size}",
            pad=20,
            fontweight="bold",
        )
        ax.set_xlabel("Epochs", fontweight="bold")
        ax.set_ylabel("Accuracy", fontweight="bold")

        # Enhanced legend
        legend = ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=False,
            shadow=False,
            borderpad=1,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

        # Add explanatory text if validation set is used
        if val_set is not None:
            plt.figtext(
                0.02,
                0.02,
                "Solid lines: training \nDashed lines: validation",
                fontsize=8,
                style="italic",
                bbox=dict(facecolor="white", alpha=0.1, edgecolor="gray", pad=2),
            )

        plt.tight_layout()
        plt.show()

        # Plot 2: Loss
        fig, ax = plt.subplots(figsize=(12, 8))

        min_loss = float("inf")
        max_loss = float("-inf")

        for model, lambda_, color in zip(models, lambda_values, colors):
            epochs_trained = len(model.list_bce)
            epochs_range = range(1, epochs_trained + 1)

            min_loss = min(min_loss, min(model.list_bce))
            max_loss = max(max_loss, max(model.list_bce))

            ax.plot(
                epochs_range,
                model.list_bce,
                color=color,
                label=f"λ={lambda_} (Train",
                alpha=alpha_train,
                linewidth=2.0,
            )

            if val_set is not None and model.list_val_bce:
                min_loss = min(min_loss, min(model.list_val_bce))
                max_loss = max(max_loss, max(model.list_val_bce))
                ax.plot(
                    epochs_range,
                    model.list_val_bce,
                    color=color,
                    linestyle="--",
                    label=f"λ={lambda_} (Val)",
                    alpha=alpha_val,
                    linewidth=1.6,
                )

        # Enhanced y-axis limits and grid
        y_padding = (max_loss - min_loss) * 0.1
        ax.set_ylim(min_loss - y_padding, max_loss + y_padding)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)  # Place grid behind data

        ax.set_title(
            f"Binary Cross Entropy Loss for Batch Size {batch_size}",
            pad=20,
            fontweight="bold",
        )
        ax.set_xlabel("Epochs", fontweight="bold")
        ax.set_ylabel("Loss", fontweight="bold")

        # Enhanced legend
        legend = ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=False,
            shadow=False,
            borderpad=1,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.9)

        # Add explanatory text if validation set is used
        if val_set is not None:
            plt.figtext(
                0.02,
                0.02,
                "Solid lines: training\nDashed lines: validation",
                fontsize=10,
                style="italic",
                bbox=dict(facecolor="white", alpha=0.1, edgecolor="gray", pad=2),
            )

        plt.tight_layout()
        plt.show()


# Call the function:
plot_lambda_comparison(
    X_train_scaled, y_train, val_set=(X_val_scaled, y_val)  # Optional validation set
)


model = LogRegClass(lr=0.01, epochs=1000, lambda_=0.0001, momentum=0.9, batch_size=128)

# Then train (fit) the model
model.fit(X_train_scaled, y_train, val_set=(X_val, y_val))  # val_set is optional

# Now you can plot the confusion matrix
model.plot_confusion_matrix(X_val_scaled, y_val)
