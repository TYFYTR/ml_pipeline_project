from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


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

def train_decision_tree(X_train, y_train, random_state=42, max_depth=None):
    """
    Trains a Decision Tree Regressor.
    Keep defaults minimal; expose random_state and max_depth for reproducibility/tuning.
    """
    model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model
