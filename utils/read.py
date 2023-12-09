import os
import cv2
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_landmarks(json_file_path):
    """
    Read landmarks.json from json file.
    :param json_file_path: json file path
    :return: landmarks.json list
    """
    with open(json_file_path) as f:
        data = json.load(f)
    return data


def read_df(file_path):
    df = pd.read_csv(file_path)
    return df


def read_data(dir_path, flatten=0, grayscale=0, resize=None, binary=0):
    """
    read data into features, labels
    :param dir_path: "../dataset/test"
    :param flatten: Bool
    :return: list: features, list: labels
    """
    features = []
    labels = []
    if not os.path.exists(dir_path):
        raise Exception("Directory does not exist!")
    else:
        class_relative_dirs = os.listdir(dir_path)
        for class_relative_dir in class_relative_dirs:
            if class_relative_dir != '.DS_Store':
                class_absolute_path = os.path.join(dir_path, class_relative_dir)
                images = os.listdir(class_absolute_path)
                for img in images:
                    img_path = os.path.join(class_absolute_path, img)
                    # Grayscale
                    if grayscale:
                        image = cv2.imread(img_path, 0)
                        # histogram equalization
                        image = cv2.equalizeHist(image)
                    else:
                        image = cv2.imread(img_path)
                    # Resize
                    if resize:
                        image = cv2.resize(image, resize)
                    # Binary
                    if binary:
                        threshold = 127.5
                        image = (image >= threshold).astype(int)
                    # Flatten
                    if flatten:
                        image = image.flatten()
                    features.append(image)
                    labels.append(class_relative_dir)
    return np.array(features), np.array(labels)


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
    # Read image data
    train_dir = "../dataset/train_binary"
    test_dir = "../dataset/test_binary"
    features, labels = read_data(train_dir, flatten=1, grayscale=1, resize=(30, 30))
    # Read data
    train_landmarks = "train_landmarks.csv"
    test_landmarks = "test_landmarks.csv"
    train = read_df(train_landmarks)
    test = read_df(test_landmarks)

