import math


class Bayes_Classifier:
    def __init__(self, feature_type, width=60, height=60, landmark_weight=1000):
        self.model = {}
        self.class_probabilities = {}
        self.feature_type = feature_type
        self.width = width
        self.height = height
        self.landmark_weight = landmark_weight
    

    def _clap(self, x, y):
        if x >= self.width:
            x = self.width - 1
        if y >= self.height:
            y = self.height - 1
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        return x, y


    def _preprocess_landmarks(self, dataset):
        # Landmarks are 0-1 normalized X, Y coordinates
        # Lets make a 200x200 grid and count the number of landmarks in each cell
        # We will use this as our feature vector
        # We then want to create a probability distribution for the probability of seeing a landmark in each cell
        
        # Dataset is a pandas DataFrame
        # Get number of unique labels
        unique_labels = dataset['label'].unique()

        for label in unique_labels:
            self.model[label] = {}
            for i in range(self.width):
                for j in range(self.height):
                    # Initialize to 1 to avoid 0 probabilities
                    self.model[label][(i, j)] = 1
        
        for idx in dataset.index:
            label = dataset["label"][idx]
            for i in range(21):
                x = int(dataset[f"keypoint{i+1}_x"][idx] * self.width)
                y = int(dataset[f"keypoint{i+1}_y"][idx] * self.height)
                x, y = self._clap(x, y)
                self.model[label][(x, y)] += self.landmark_weight
        
        for label in unique_labels:
            total = 0
            for i in range(self.width):
                for j in range(self.height):
                    total += self.model[label][(i, j)]
            for i in range(self.width):
                for j in range(self.height):
                    self.model[label][(i, j)] /= total
        
        for label in unique_labels:
            self.class_probabilities[label] = 0
        for idx in dataset.index:
            label = dataset["label"][idx]
            self.class_probabilities[label] += 1
        for label in unique_labels:
            self.class_probabilities[label] /= len(dataset)
        return


    def _preprocess_pixels(self, dataset):
        features, labels = dataset[0], dataset[1]
        unique_labels = list(set(labels))
        feature_size = len(features[0])
        for label in unique_labels:
            self.model[label] = [1] * feature_size
        for idx in range(len(features)):
            label = labels[idx]
            for i in range(feature_size):
                self.model[label][i] += features[idx][i]
        for label in unique_labels:
            total = sum(self.model[label])
            for i in range(feature_size):
                self.model[label][i] /= total
        for label in unique_labels:
            self.class_probabilities[label] = 0
        for idx in range(len(features)):
            label = labels[idx]
            self.class_probabilities[label] += 1
        for label in unique_labels:
            self.class_probabilities[label] /= len(features)
        return
    

    def _classify_pixels(self, dataset):
        features, labels = dataset[0], dataset[1]
        preds = []
        for idx in range(len(features)):
            ps = {}
            for label in self.model:
                p = 1
                for i in range(len(features[idx])):
                    p += math.log(self.model[label][i]) * features[idx][i]
                p *= self.class_probabilities[label]
                ps[label] = p
            max_label = max(ps, key=ps.get)
            preds.append(max_label)
        return preds
    

    def _classify_landmarks(self, dataset):
        unique_labels = dataset['label'].unique()

        for idx in dataset.index:
            ps = {}
            for label in unique_labels:
                p = 1
                for i in range(21):
                    x = int(dataset[f"keypoint{i+1}_x"][idx] * self.width)
                    y = int(dataset[f"keypoint{i+1}_y"][idx] * self.height)
                    x, y = self._clap(x, y)
                    p += math.log(self.model[label][(x, y)])
                p *= self.class_probabilities[label]
                ps[label] = p
            max_label = max(ps, key=ps.get)
            dataset.at[idx, "predicted_label"] = max_label
        return dataset


    def train(self, dataset):
        if self.feature_type == "pixels":
            self._preprocess_pixels(dataset)
        elif self.feature_type == "landmarks":
            self._preprocess_landmarks(dataset)
        else:
            raise ValueError("Invalid feature type")
        
        return


    def classify(self, dataset):
        if self.feature_type == "pixels":
            preds = self._classify_pixels(dataset)
        elif self.feature_type == "landmarks":
            preds = self._classify_landmarks(dataset)
        else:
            raise ValueError("Invalid feature type")
        
        return preds
    
