from src.preprocessing import load_california_data, make_features_and_target, split_train_test
from src.models import train_linear_regression, predict
from src.evaluation import evaluate_regression, plot_actual_vs_predicted




def main():
    # 1. Load full dataset
    df = load_california_data()

    # 2. Split into features and target
    X, y = make_features_and_target(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 4. Train the model
    model = train_linear_regression(X_train, y_train)

    # 5. Make predictions on the test set
    y_pred = predict(model, X_test)

    # 6. Evaluate the model
    metrics = evaluate_regression(y_test, y_pred)

    print("Evaluation metrics for Linear Regression:")
    print(f"  MSE : {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}")

    # Visualize predictions vs actual values
    plot_actual_vs_predicted(y_test, y_pred)



if __name__ == "__main__":
    main()
