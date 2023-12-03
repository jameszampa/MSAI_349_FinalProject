import math
import numpy as np
from sklearn.decomposition import PCA


def dimension_scaling(features):
    """
        Scales the features using z-score normalization.

        Parameters:
        - features (list of lists): List of feature vectors to be scaled.

        Returns:
        - list of lists: Scaled feature vectors.
        """
    for f in features:
        mean = sum(f)/len(f)
        variance = sum((x - mean) ** 2 for x in f) / len(f)
        std_dev = math.sqrt(variance)
        f = [(x-mean)/std_dev for x in f]
    return features


class DimensionReduction:

    def __init__(self, features, n_component=None):
        self.pca = PCA(svd_solver='randomized', n_components=n_component).fit(features)
        self.component_variance = np.cumsum(self.pca.explained_variance_ratio_)

    def pca_transform(self, features):
        """
        Transform the features to reduced dimensionality
        :param n_component: number of dimension after reduction
        :param features: numpy.array: shape: (number of features, feature dimension), suggested dimension: <2500,
        otherwise it may take a long execution time
        :return: features with reduced dimensionality
        """
        return self.pca.transform(features)

    def get_variance_fraction_by_n_component(self, n_component):
        """
        Evaluate how representative is the number of n components of the original features, range is 0-1
        :param n_component: number of dimension after reduction
        :return: percentage: the percentage of explained variance, range is 0-1
        """
        return round(self.component_variance[n_component - 1], 2)

    def get_n_component_by_variance_threshold(self, variance_threshold):
        """
        Given an ideal variance threshold, find the minimal n_component
        :param variance_threshold: the percentage of explained variance, range is 0-1
        :return: int: n_component
        """
        return np.argmax(self.component_variance > variance_threshold)

    @staticmethod
    def choose_n_component(features):
        """
        This function helps to choose the suitable amount of n_component(dimensions)
        The more cumulative explained variance the better, but we also need to find a trade-off
        between dimension and variance.
        Example: features: (27352, 2500), According to the visualize_dimension_reduction notebook
        1) n_components = [6, 9, 16, 35, 102, 588] explains [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        of variance
        2) The reconstruction of the images shows that we need at least 102 dimensions(0.95 of variance) to recognize
        the gestures with human eyes

        :param features: numpy.array: shape: (number of features, feature dimension), suggested dimension: <2500,
        otherwise it may take a long execution time
        :return: numpy.array: [cumulative explained variance]
        """
        pca = PCA(svd_solver='randomized').fit(features)
        return np.cumsum(pca.explained_variance_ratio_)


if __name__ == "__main__":
    from utils.read import read_data
    train_features, train_labels = read_data('../dataset/train', flatten=1, grayscale=1, resize=(50, 50))
    train_features = np.array(train_features)
    pca = DimensionReduction(train_features,None)
    percentage_of_variance = pca.get_variance_fraction_by_n_component(102)
    test_n_component = pca.get_n_component_by_variance_threshold(0.9)
    print (percentage_of_variance, test_n_component)
    projected = pca.pca_transform(train_features)


def run_preprocessing():
    from utils.read import read_data
    train_features, train_labels = read_data('/Users/jeongyoon/Desktop/GitBlog/MSAI_349_FinalProject-1/dataset/train_binary', flatten=1, grayscale=1, resize=(28, 28))
    train_features = np.array(train_features)
    test_features, test_labels  =read_data('/Users/jeongyoon/Desktop/GitBlog/MSAI_349_FinalProject-1/dataset/test_binary', flatten=1, grayscale=1, resize=(28, 28))
    test_features = np.array(test_features)
    
    pca = DimensionReduction(train_features,None)
    percentage_of_variance = pca.get_variance_fraction_by_n_component(102)
    test_n_component = pca.get_n_component_by_variance_threshold(0.9)
    print (percentage_of_variance, test_n_component)
    train_pca = pca.pca_transform(train_features)
    test_pca = pca.pca_transform(test_features)
    return train_pca, test_pca , train_labels , test_labels





