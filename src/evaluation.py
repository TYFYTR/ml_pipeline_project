from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def evaluate_regression(y_true, y_pred):
    """
    Evaluates a regression model using common metrics.
    Returns a dictionary with MSE, RMSE, MAE, and R2.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Scatter plot of actual vs predicted values.
    Helps you see how close predictions are to the ideal y = x line.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Actual vs Predicted")
    
    # Ideal line: perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred):

    """ Plot residuals (errors) vs predicted values.
    Residual = actual - predicted.
    A good model has residuals roughly centered around 0 with no clear pattern.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals (actual - predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()


def plot_both(y_true, y_pred):
    """
    Displays Actual vs Predicted and Residuals side-by-side.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "--")
    axes[0].set_xlabel("Actual values")
    axes[0].set_ylabel("Predicted values")
    axes[0].set_title("Actual vs Predicted")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Predicted values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    plt.show()
