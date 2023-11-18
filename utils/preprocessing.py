import math


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
