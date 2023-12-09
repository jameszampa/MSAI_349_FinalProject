import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.evaluate import evaluate
from utils.read import read_df, read_data
from utils.preprocessing import DimensionReduction, get_color_histogram
from model.decision_tree import ID3, Prune
from model.random_forest import RandomForest
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


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
        print("Preprocessing")
        # Scale
        X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1] / 2), 2)
        key_points_scaled = []
        for key_point in X_train:
            # Scale the key points in each picture
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
    def load_landmarks():
        # Load data & preprocess
        print("Load data")
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
        return X_train, y_train, X_test, y_test

    @staticmethod
    def test_landmarks(criterion='information_gain', max_feature_number=None):
        # Load data & preprocess
        X_train, y_train, X_test, y_test = CustomDecisionTree.load_landmarks()
        # Train ID3
        print("Train the model")
        tree = ID3.ID3(X_train, criterion=criterion, max_feature_number=max_feature_number)
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
        # Load data & preprocess
        X_train, y_train, X_test, y_test = CustomDecisionTree.load_landmarks()
        # Train
        forest = RandomForest.random_forest(X_train, n_tree=10)
        # Test
        predictions = []
        for test_feature in X_test:
            pred = RandomForest.random_forest_evaluate(forest, test_feature)
            predictions.append(pred)
        # Evaluate
        precision, recall, f1 = evaluate('random_forest', y_test, predictions)


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
    def test_landmarks_max_feature():
        # Bagged trees tuning by varying number of estimators for bagging.
        # Load data & preprocess
        X_train, y_train, X_test, y_test = CustomDecisionTree.load_landmarks()

        max_f_no_list = [10, 20, 30, 50, 100, 200, 300]
        # max_f_no_list = [10]
        df1 = pd.DataFrame(index=range(len(max_f_no_list)),
                           columns=["max_features_number", "f1"])
        for j in range(len(max_f_no_list)):
            print(j)
            start = time.time()
            # Train ID3
            print("Train the model")
            tree = ID3.ID3(X_train, criterion='information_gain', max_feature_number=max_f_no_list[j])
            # Test
            print("Make predictions on the test set")
            predictions = []
            for test_feature in X_test:
                pred = ID3.evaluate(tree, test_feature)
                predictions.append(pred)
            # Evaluate
            print ("Evaluate the accuracy")
            precision, recall, f1 = evaluate('id3', y_test, predictions)
            print (precision, recall, f1)
            end = time.time()
            print(end - start)
            df1.loc[j] = [max_f_no_list[j], f1]
        plt.plot(df1.loc[0:6]["max_features_number"], df1.loc[0:6]["f1"], color='steelblue',
                 marker='.', linestyle='dotted', label="max_features_number")
        plt.xlabel("Max Features Number")
        plt.ylabel("F1")
        plt.legend()
        plt.show()
        return df1


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
        # Grayscale, Histogram equalization, Resize in the data reading process
        X_train, y_train = read_data('../dataset/train', flatten=1, grayscale=1, resize=(50, 50))
        X_test, y_test = read_data('../dataset/test', flatten=1, grayscale=1, resize=(50, 50))

        # Preprocessing
        print("Preprocessing")
        # Scale
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # PCA
        pca = DimensionReduction(X_train, 588)
        X_train = pca.pca_transform(X_train)

        # Train
        print("Training")
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X_train, y_train)

        # Evaluate
        print("Evaluate")
        # Predict the response for test dataset
        X_test = scaler.transform(X_test)
        X_test = pca.pca_transform(X_test)
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Evaluate the accuracy")
        precision, recall, f1 = evaluate('sklearn_landmarks', y_test, y_pred)


    @staticmethod
    def test_histogram_sklearn():
        print("Load a dataset")
        # Grayscale, Histogram equalization, Resize in the data reading process
        X_train, y_train = read_data('../dataset/train', flatten=0, grayscale=0, resize=(50, 50))
        X_test, y_test = read_data('../dataset/test', flatten=0, grayscale=0, resize=(50, 50))

        # Preprocessing
        print("Preprocessing")
        X_train = get_color_histogram(X_train, 0)
        X_test = get_color_histogram(X_test, 0)

        # Train
        print("Training")
        clf = DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X_train, y_train)

        # Evaluate
        print("Evaluate")
        # Predict the response for test dataset
        y_pred = clf.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Evaluate the accuracy")
        precision, recall, f1 = evaluate('decisiontree_sklearn_histogram', y_test, y_pred)

    @staticmethod
    def test_bagging_tree():
        # Bagged trees tuning by varying number of estimators for bagging.
        # Load dataset
        print("Load a dataset")
        # Grayscale, Histogram equalization, Resize in the data reading process
        X_train, y_train = read_data('../dataset/train', flatten=1, grayscale=1, resize=(50, 50))
        X_test, y_test = read_data('../dataset/test', flatten=1, grayscale=1, resize=(50, 50))

        print("Train")
        # Estimator
        estimators = [10, 20, 30, 50]
        num_estimators = len(estimators)
        # df1 = pd.DataFrame(index=range(num_estimators),
        #                    columns=["estimators", "accuracy"])
        dt = DecisionTreeClassifier()
        for j in range(num_estimators):
            print(j)
            bg = BaggingClassifier(
                base_estimator=dt, n_estimators=estimators[j], random_state=0)
            bg.fit(X_train, y_train)
            # accuracy = bg.score(X_test, y_test)
            y_pred = bg.predict(X_test)
            print("Evaluate the accuracy")
            precision, recall, f1 = evaluate('decisiontree_sklearn_histogram', y_test, y_pred)
            print (precision, recall, f1)
            # df1.loc[j] = [estimators[j], accuracy]
        # return df1

    @staticmethod
    def plot_bagging(df):
        '''Plot number of estimators vs accuracy with different number of leaf nodes
        for the underlying decision tree note that the lines for 500 leaf nodes and
        800 leaf nodes completely overlap each other.'''
        # plt.plot(df.loc[0:3]["estimators"], df.loc[0:3]["accuracy"], color='steelblue',
        #          marker='.', linestyle='dotted', label="Bagging")
        # plt.plot(df.loc[4:7]["estimators"], df.loc[4:7]["accuracy"], color='green',
        #          marker='.', linestyle='dotted', label="Bagging w/500 Leaf Nodes")
        # plt.plot(df.loc[8:11]["estimators"], df.loc[8:11]["accuracy"], color='red',
        #          marker='.', linestyle='dotted', label="Bagging w/800 Leaf Nodes")
        plt.xlabel("Number of Estimators")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        return


if __name__ == '__main__':
    start = time.time()
    max_f_no = 10
    CustomDecisionTree.test_landmarks(criterion='information_gain', max_feature_number=max_f_no)
    end = time.time()
    print(end-start)
    # LibraryDecisionTree.test_image_sklearn_histogram()
    # BAGGING_DF = LibraryDecisionTree.test_bagging_tree()
    # LibraryDecisionTree.plot_bagging(BAGGING_DF)
    # CustomDecisionTree.test_landmarks_max_feature()


