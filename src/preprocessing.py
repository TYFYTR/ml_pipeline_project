from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split


def load_california_data():
    """
    Loads the California housing dataset and returns it as a pandas DataFrame.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

def make_features_and_target(df):
    """
    Splits the full DataFrame into:
    - X: features (all columns except MedHouseVal)
    - y: target (the MedHouseVal column)
    """
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into train and test sets.
    - test_size=0.2 means 20% of the data is for testing
    - random_state=42 makes the split reproducible
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
