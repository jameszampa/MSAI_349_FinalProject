import numpy as np
from utils.read import read_df
from utils.read import read_data
from utils.evaluate import evaluate
from utils.preprocessing import DimensionReduction
from model.decision_tree import ID3, Prune
from model.random_forest import RandomForest
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def preprocess_image(features):
    # Preprocessing
    print("Preprocessing")
    # Scale
    scaler = StandardScaler()
    scaler.fit(features)
    X_train = scaler.transform(features)
    # PCA
    pca = DimensionReduction(X_train, 588)
    X_train = pca.pca_transform(X_train)
    # Discretize the PCA-transformed features
    n_bins = 100
    k_bins_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    data_discretized = k_bins_discretizer.fit_transform(X_train)
    return data_discretized


class CustomDecisionTree:

    @staticmethod
    def preprocess_landmarks(X_train):
        print("Preprocessing}")
        # Scale
        X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / 2), 2)
        key_points_scaled = []
        for key_point in X_train:
            # Scale the keypoints in each picture
            scaler = MinMaxScaler()
            key_point_scaled = scaler.fit_transform(key_point)
            key_points_scaled.append(key_point_scaled.flatten())
        # Create 10 evenly spaced bins between 0 and 1, discretize the points between 0-1
        key_points_scaled = np.round(np.array(key_points_scaled), 5)
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        inds = np.digitize(key_points_scaled, bins, right=True)
        # One-hot encoding
        key_points_one_hot = []
        for key_point in inds:
            x = np.eye(np.max(key_point) + 1)[key_point]
            key_points_one_hot.append(x.flatten())
        return np.array(key_points_one_hot)

    @staticmethod
    def test_landmarks():
        # load data & preprocess
        train_df = read_df('../dataset/train_landmarks.csv')
        test_df = read_df('../dataset/test_landmarks.csv')
        X_train = CustomDecisionTree.preprocess_landmarks(train_df.values[:, 1:]).astype('int')
        y_train = train_df.values[:, 0]
        X_test = CustomDecisionTree.preprocess_landmarks(test_df.values[:, 1:]).astype('int')
        y_test = test_df.values[:, 0]

        # Prepare data structure
        train_columns = [train_df.columns[0]]
        train_columns.extend(np.arange(0, len(X_train[0])))
        test_columns = np.arange(0, len(X_train[0]))
        X_train = np.concatenate((np.expand_dims(y_train, axis=1), X_train), axis=1)
        X_train = [dict(zip(train_columns, row)) for row in X_train]
        X_test = [dict(zip(test_columns, row)) for row in X_test]

        # Train ID3
        print("Train the model")
        tree = ID3.ID3(X_train)
        # Test
        print("Make predictions on the test set")
        predictions = []
        for test_feature in X_test:
            pred = ID3.evaluate(tree, test_feature)
            predictions.append(pred)
        # Evaluate
        print ("Evaluate the accuracy")
        precision, recall, f1 = evaluate('id3', y_test, predictions)

    @staticmethod
    def test_image():
        print("Load a dataset")
        # Grayscale, Histogram equalization, Resize in the data reading process
        X_train, y_train = read_data('../dataset/train_binary', flatten=1, grayscale=1, resize=(30, 30), binary=1)
        X_test, y_test = read_data('../dataset/test_binary', flatten=1, grayscale=1, resize=(30, 30), binary=1)

        print("Preprocessing")
        # X_train = get_color_histogram(X_train)
        # X_test = get_color_histogram(X_test)
        # X_train
        # Scale
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # PCA
        # pca = DimensionReduction(X_train, 588)
        # X_train = pca.pca_transform(X_train)
        # Discretize the PCA-transformed features
        # n_bins = 100
        # k_bins_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        # X_train = k_bins_discretizer.fit_transform(X_train)

        # X_test
        # Scale
        # X_test = scaler.transform(X_test)
        # PCA
        # X_test = pca.pca_transform(X_test)
        # Discretize the PCA-transformed features
        # n_bins = 100
        # k_bins_discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        # X_test = k_bins_discretizer.fit_transform(X_test)

        # Prepare data
        train_columns = ['label']
        train_columns.extend(np.arange(0, len(X_train[0])))
        test_columns = np.arange(0, len(X_test[0]))
        X_train = np.concatenate((np.expand_dims(y_train, axis=1), X_train), axis=1)
        X_train = [dict(zip(train_columns, row)) for row in X_train]
        X_test = [dict(zip(test_columns, row)) for row in X_test]

        # Train
        print("Train the model")
        tree = ID3.ID3(X_train)

        # Test
        print("Make predictions on the test set")
        predictions = []
        for test_feature in X_test:
            pred = ID3.evaluate(tree, test_feature)
            predictions.append(pred)
        # Evaluate
        print ("Evaluate the accuracy")
        precision, recall, f1 = evaluate('id3', y_test, predictions)

    @staticmethod
    def test_landmarks_pruning():

        # load data & preprocess
        train_df = read_df('../dataset/train_landmarks.csv')
        test_df = read_df('../dataset/test_landmarks.csv')
        X_train = CustomDecisionTree.preprocess_landmarks(train_df.values[:, 1:]).astype('int')
        y_train = train_df.values[:, 0]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_test = CustomDecisionTree.preprocess_landmarks(test_df.values[:, 1:]).astype('int')
        y_test = test_df.values[:, 0]

        # Prepare data structure
        train_columns = [train_df.columns[0]]
        train_columns.extend(np.arange(0, len(X_train[0])))
        test_columns = np.arange(0, len(X_train[0]))
        X_train = np.concatenate((np.expand_dims(y_train, axis=1), X_train), axis=1)
        X_train = [dict(zip(train_columns, row)) for row in X_train]
        X_test = [dict(zip(test_columns, row)) for row in X_test]
        X_val = np.concatenate((np.expand_dims(y_val, axis=1), X_val), axis=1)
        X_val = [dict(zip(train_columns, row)) for row in X_val]

        # Train ID3
        print("Train the model")
        tree = ID3.ID3(X_train)
        # Pruning
        print("Pruning")
        Prune.prune(tree, X_val)
        # Test
        print("Make predictions on the test set")
        predictions = []
        for test_feature in X_test:
            pred = ID3.evaluate(tree, test_feature)
            predictions.append(pred)
        # Evaluate
        print("Evaluate the accuracy")
        precision, recall, f1 = evaluate('id3', y_test, predictions)

    @staticmethod
    def test_landmarks_random_forest():
        # load data & preprocess
        train_df = read_df('../dataset/train_landmarks.csv')
        test_df = read_df('../dataset/test_landmarks.csv')
        X_train = CustomDecisionTree.preprocess_landmarks(train_df.values[:, 1:]).astype('int')
        y_train = train_df.values[:, 0]
        X_test = CustomDecisionTree.preprocess_landmarks(test_df.values[:, 1:]).astype('int')
        y_test = test_df.values[:, 0]

        # Prepare data structure
        train_columns = [train_df.columns[0]]
        train_columns.extend(np.arange(0, len(X_train[0])))
        test_columns = np.arange(0, len(X_train[0]))
        X_train = np.concatenate((np.expand_dims(y_train, axis=1), X_train), axis=1)
        X_train = [dict(zip(train_columns, row)) for row in X_train]
        X_test = [dict(zip(test_columns, row)) for row in X_test]

        # Train
        forest = RandomForest.random_forest(X_train, n_tree=10)
        # Test
        predictions = []
        for test_feature in X_test:
            pred = RandomForest.random_forest_evaluate(forest, test_feature)
            predictions.append(pred)
        # Evaluate
        precision, recall, f1 = evaluate('random_forest', y_test, predictions)


class LibraryDecisionTree:

    @staticmethod
    def test_landmarks_sklearn():
        # Load data
        print("Load datasets")
        train_df = read_df('../dataset/train_landmarks.csv')
        test_df = read_df('../dataset/test_landmarks.csv')
        X_train = train_df.values[:, 1:]
        y_train = train_df.values[:, 0]
        X_test = test_df.values[:, 1:]
        y_test = test_df.values[:, 0]
        X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / 2), 2)
        key_points_scaled = []
        for key_point in X_train:
            # Scale the keypoints in each picture
            scaler = MinMaxScaler()
            key_point_scaled = scaler.fit_transform(key_point)
            key_points_scaled.append(key_point_scaled.flatten())

        # Train
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(key_points_scaled, y_train)
        # Predict the response for test dataset
        X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1] / 2), 2)
        test_key_points_scaled = []
        for key_point in X_test:
            # Scale the keypoints in each picture
            scaler = MinMaxScaler()
            key_point_scaled = scaler.fit_transform(key_point)
            test_key_points_scaled.append(key_point_scaled.flatten())
        y_pred = clf.predict(test_key_points_scaled)
        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        # Evaluate
        print("Evaluate the accuracy")
        precision, recall, f1 = evaluate('sklearn_landmarks', y_test, y_pred)

    @staticmethod
    def test_image_sklearn():
        print("Load a dataset")
        X_train, y_train = read_data('../dataset/train', flatten=1, grayscale=1, resize=(50, 50))
        X_test, y_test = read_data('../dataset/test', flatten=1, grayscale=1, resize=(50, 50))

        # Preprocessing
        print("Preprocessing")
        # Grayscale, Histogram equalization, Resize in the data reading process
        # Scale
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # PCA
        # pca = DimensionReduction(X_train, 102)
        # X_train = pca.pca_transform(X_train)

        # Train
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X_train, y_train)

        # Predict the response for test dataset
        X_test_t = scaler.transform(X_test)
        # X_test_t = pca.pca_transform(X_test_t)
        y_pred = clf.predict(X_test_t)

        # Evaluate
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Evaluate the accuracy")
        precision, recall, f1 = evaluate('sklearn_landmarks', y_test, y_pred)


if __name__ == '__main__':
    # X_train, y_train = read_data('../dataset/train', flatten=0, grayscale=1, resize=(50, 50))
    # get_color_histogram(X_train)
    CustomDecisionTree.test_landmarks()
