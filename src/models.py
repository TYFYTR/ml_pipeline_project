from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    """
    Trains a simple Linear Regression model on the training data.
    Returns the trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Uses the trained model to make predictions on the test set.
    """
    y_pred = model.predict(X_test)
    return y_pred