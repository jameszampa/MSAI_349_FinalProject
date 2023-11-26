import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_mean = {}
        self.class_variance = {}
        self.class_prior = {}

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        for c in self.classes:
            X_c = self._get_x_c(X, y, c)
            self.class_mean[str(c)] = np.mean(X_c, axis=0)
            self.class_variance[str(c)] = np.var(X_c, axis=0)
            self.class_prior[str(c)] = X_c.shape[0] / self.n_samples


    def _get_x_c(self, X, Y, c):
        X_c = []
        for x, y in zip(X, Y):
            if y == c:
                X_c.append(x)
        return np.array(X_c)


    def predict(self, X):
        return [self._predict(x) for x in X]
    
    
    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.class_prior[str(c)])
            class_conditional = np.sum(np.log(self._pdf(x, c)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    

    def _pdf(self, x, c):
        mean = self.class_mean[str(c)]
        var = self.class_variance[str(c)]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator