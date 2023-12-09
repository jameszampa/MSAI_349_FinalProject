import math
import numpy as np
from utils.evaluate import evaluate
from utils.read import read_features_labels



class Helper:

    @staticmethod
    def euclidean(a, b):
        """
        :param a: an n-dimensional vector with shape (n, )
        :param b: an n-dimensional vector with shape (n, )
        :return: a scalar float value of the Euclidean distance between vectors a and b
        """
        if len(a) != len(b):
            raise Exception("The dimensions of a and b do not match!")
        dist_sum = 0
        for i in range(len(a)):
            dist_sum += (a[i] - b[i]) ** 2
        dist = math.sqrt(dist_sum)
        return dist

    @staticmethod
    def get_norm(a):
        """
        :param a: an n-dimensional vector with shape (n, )
        :return: norm of a
        """
        norm_sum = 0
        for i in range(len(a)):
            norm_sum += a[i] ** 2
        return math.sqrt(norm_sum)

    @staticmethod
    def dot_product(a, b):
        """
        :param a: an n-dimensional vector with shape (n, )
        :param b: an n-dimensional vector with shape (n, )
        :return: dot product of a and b
        """
        product = 0
        for i in range(len(a)):
            product += a[i] * b[i]
        return product

    @staticmethod
    def cosim(a, b):
        """
        :param a: an n-dimensional vector with shape (n, )
        :param b: an n-dimensional vector with shape (n, )
        :return: a scalar float value of the cosine similarity between vectors a and b
        """
        if len(a) != len(b):
            raise Exception("The dimensions of a and b do not match!")
        norm_a = Helper.get_norm(a)
        norm_b = Helper.get_norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dist = Helper.dot_product(a, b) / (norm_a * norm_b)
        return dist


class KNearestNeighbor:

    def __init__(self, n_neighbors, distance_measure, aggregator):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        if self.aggregator not in ['mode', 'mean', 'median']:
            raise ValueError("Not a valid argument for aggregator")

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        self.train_features = features if isinstance(features, np.ndarray) else np.array(features)
        self.targets = targets if isinstance(targets, np.ndarray) else np.array(targets)

    def find_aggregator(self, nearest_neighbours_targets, neighbour_distances):
        """
            Aggregate the target values of nearest neighbours based on the chosen aggregator method.

            Parameters:
            - nearest_neighbours_targets (numpy.ndarray): Targets of the nearest neighbours. Can be single or multi-dimensional.
            - neighbour_distances (numpy.ndarray): Distances of the nearest neighbours to the query instance.

            Returns:
            - numpy.ndarray: Aggregated target values.

            Notes:
            - 'mode' aggregator uses weighted mode in case of ties.
            - Weights for distances are computed with a small constant to prevent division by zero.
            """
        results = []
        # To Handle Multi dimensional targets
        if len(nearest_neighbours_targets.shape) == 1:
            nearest_neighbours_targets = nearest_neighbours_targets.reshape(-1, 1)

        # Added 0.1 in denominator to avoid division by zero
        weights = 1 / (neighbour_distances + 0.1)

        if self.aggregator == 'mode':
            for k in range(nearest_neighbours_targets.shape[1]):
                unique_values, counts = np.unique((nearest_neighbours_targets[:, k]), return_counts=True)
                if len(unique_values) == 1:
                    results.append(unique_values[0])
                else:
                    max_count = max(counts)
                    modes = unique_values[counts == max_count]
                    if len(modes) == 1:
                        results.append(modes[0])
                    else:
                        weighted_counts = {}
                        for mode in modes:
                            weighted_counts[mode] = sum(weights[nearest_neighbours_targets[:, k] == mode])
                        results.append(max(weighted_counts, key=weighted_counts.get))

            return np.array(results)
        if self.aggregator == 'mean':
            for k in range(nearest_neighbours_targets.shape[1]):
                mean_value = np.mean(nearest_neighbours_targets[:, k])
                results.append(int(round(mean_value)))
            return np.array(results)

        if self.aggregator == 'median':
            for k in range(nearest_neighbours_targets.shape[1]):
                median_value = np.median(nearest_neighbours_targets[:, k])
                results.append(int(round(median_value)))
            return np.array(results)

    def find_n_nearest(self, test_feature, ignore_first):
        """
            Find the n nearest neighbors for the given test feature from the training set.

            Parameters:
            - test_feature (list/np.array): The feature vector of the test instance.
            - ignore_first (str): If 'True', the function will ignore the closest neighbor (useful for leave-one-out cross-validation).

            Returns:
            - tuple: A tuple containing two numpy arrays:
              1. Indices of the nearest neighbors in the training set.
              2. Distances of the nearest neighbors to the test feature.

            Notes:
            - Supports both 'euclidean' and 'cosim' (cosine similarity) as distance measures.
            """

        if ignore_first:
            nearest_neighbors_count = self.n_neighbors + 1
        else:
            nearest_neighbors_count = self.n_neighbors

        nearest_neighbours = [(float('inf'), None) for _ in range(nearest_neighbors_count)]

        for j in range(len(self.train_features)):
            if self.distance_measure == 'euclidean':
                distance = Helper.euclidean(test_feature, self.train_features[j])
            if self.distance_measure == 'cosim':
                distance = 1-Helper.cosim(test_feature, self.train_features[j])
            if distance < max(a for (a, b) in nearest_neighbours):
                nearest_neighbours[-1] = (distance, j)
                nearest_neighbours.sort()

        neighbour_indices = [b for (a, b) in nearest_neighbours]
        neighbour_distances = [a for (a, b) in nearest_neighbours]

        if ignore_first:
            neighbour_indices = neighbour_indices[1:]

        return np.array(neighbour_indices), np.array(neighbour_distances)

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).
3
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor.
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """

        predicted_targets = []
        for i in range(len(features)):
            print("feature,",i)
            nearest_indices, neighbour_distances = self.find_n_nearest(features[i], ignore_first)
            predicted_target = self.find_aggregator(self.targets[nearest_indices], neighbour_distances)
            predicted_targets.append(predicted_target)
        return np.array(predicted_targets)


if __name__ == '__main__':
    # Load dataset
    train_features, train_labels = read_features_labels('../dataset/train_landmarks.csv')
    test_features, test_labels = read_features_labels('../dataset/test_landmarks.csv')
    # Train
    metric = 'cosim'
    classifier = KNearestNeighbor(n_neighbors=3, distance_measure=metric, aggregator="mode")
    classifier.fit(train_features, train_labels)
    # Test
    print("in test")
    predictions_list=classifier.predict(test_features)
    predictions = [i[0] for i in predictions_list]
    # Evaluate
    evaluate('knn', test_labels, predictions)

