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

    @staticmethod
    def pca(n_component, features):
        """
        :param n_component: number of dimension after reduction
        :param features: numpy.array: shape: (number of features, feature dimension), suggested dimension: <2500,
        otherwise it may take a long execution time
        :return:
        """
        return PCA(svd_solver='randomized', n_components=n_component).fit_transform(features)

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
    projected = DimensionReduction.pca(102, train_features)













