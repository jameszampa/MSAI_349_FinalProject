import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.read import read_features_labels


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
    def dimension_reduction(n_component, features):
        """
        :param n_component: number of dimension after reduction
        :param features: numpy.array: shape: (number of features, feature dimension)
        :return:
        """
        pca = PCA(n_component)
        projected = pca.fit_transform(features)
        return projected

    @staticmethod
    def choose_n_component():
        """
        This function helps to choose the suitable amount of n_component(dimensions)
        The more cumulative explained variance the better, but we need to find a trade off
        between dimension and variance.
        Note: x-axis maximum = min(feature number, feature dimension)
        :return: dict: {number of component: cumulative explained variance}
        """
        pca = PCA().fit(train_features)  # n=21 0.80; n=57, 0.9; n=111, 0.95; n=284, 0.99
        x = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(x)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig('pca.png')
        return {index: value for index, value in enumerate(x)}

    @staticmethod
    def visualize_top_2_dimension(features_after_pca, labels):
        """
        Visualize top 2(most representative) dimensions of the features after pca
        to get an idea of the data. TODO: Is there software to see more dimensions?
        """
        plt.scatter(features_after_pca[:, 0], features_after_pca[:, 1],
                    c=np.array(labels), edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('Accent', 10))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.savefig('component.png')


if __name__ == "__main__":
    train_features, train_labels = read_features_labels('../dataset/sign_mnist_train.csv')
    train_features = np.array(train_features)
    projected = DimensionReduction.dimension_reduction(n_component=57, features=train_features)
    DimensionReduction.visualize_top_2_dimension(projected, train_labels)













