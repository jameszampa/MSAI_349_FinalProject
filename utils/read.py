import json
import pandas as pd
from sklearn.model_selection import train_test_split


def read_landmarks(json_file_path):
    """
    Read landmarks from json file.
    :param json_file_path: json file path
    :return: landmarks list
    """
    with open(json_file_path) as f:
        data = json.load(f)
    return data


def read_df(file_path):
    df = pd.read_csv(file_path)
    return df


def read_features_labels(file_path):
    """
    read the data and split into feature list and label list
    :param file_path: file_path
    :return: list1: feature list [[...], [...], [...], ...]; list2: label list [...]
    """
    df = pd.read_csv(file_path)
    data = [row[1:] for row in df.values]
    label = [row[0] for row in df.values]
    return data, label


def split_dataset(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and validation sets.

    Parameters:
    - X: Features of the dataset.
    - y: Labels of the dataset.
    - test_size: The proportion of the dataset to include in the validation set (default is 0.2).
    - random_state: Seed for the random number generator (optional).

    Returns:
    - X_train, X_val, y_train, y_val: Split datasets for training and validation.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    train_path = '../dataset/sign_mnist_train.csv'
    test_path = '../dataset/sign_mnist_test.csv'
    # Read data
    train = read_df(train_path)
    test = read_df(test_path)
    # Read and split into features and labels
    train_features, train_labels = read_features_labels(train_path)
    test_features, test_labels = read_features_labels(test_path)
