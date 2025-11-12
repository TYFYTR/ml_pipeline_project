from src.preprocessing import load_california_data, make_features_and_target, split_train_test


def main():
    # 1. Load full dataset
    df = load_california_data()

    # 2. Split into features and target
    X, y = make_features_and_target(df)

    # 3. Split into train/test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 4. Quick sanity checks
    print("Full data shape:", df.shape)
    print("X (features) shape:", X.shape)
    print("y (target) shape:", y.shape)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)


if __name__ == "__main__":
    main()
