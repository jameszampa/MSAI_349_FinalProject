import math


class Bayes_Classifier:
    def __init__(self, feature_type, width=250, height=250, prob_type="individual", kernel_size=9, kernel_decay_method="distance"):
        self.model = {}
        self.class_probabilities = {}
        self.feature_type = feature_type
        self.width = width
        self.height = height
        self.landmark_weight = self.width * self.height
        # Individual: Creates individual probability distributions for each keypoint
        # Collective: Creates a collective probability distribution for all keypoints
        self.prob_type = prob_type
        self.kernel_size = kernel_size
        # Distance: Kernel weights are inversely proportional to distance from center
        # None: same as center weight
        self.kernel_decay_method = kernel_decay_method
    

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


    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


    def _preprocess_landmarks_collective(self, dataset):
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
                if self.kernel_size > 1:
                    # kernel_size centered around (x, y)
                    for j in range(-self.kernel_size // 2, self.kernel_size // 2 + 1):
                        for k in range(-self.kernel_size // 2, self.kernel_size // 2 + 1):
                            x1 = x + j
                            y1 = y + k
                            x1, y1 = self._clap(x1, y1)
                            if x1 == x and y1 == y:
                                continue
                            if self.kernel_decay_method == "distance":
                                self.model[label][(x1, y1)] += self.landmark_weight / self._dist((x, y), (x1, y1))
                            else:
                                self.model[label][(x1, y1)] += self.landmark_weight
        
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
    
    def _preprocess_landmarks_individual(self, dataset):
        unique_labels = dataset['label'].unique()
        for label in unique_labels:
            self.model[label] = {}
            for i in range(21):
                self.model[label][i] = {}
                for j in range(self.width):
                    for k in range(self.height):
                        # Initialize to 1 to avoid 0 probabilities
                        self.model[label][i][(j, k)] = 1
        
        for idx in dataset.index:
            label = dataset["label"][idx]
            for i in range(21):
                x = int(dataset[f"keypoint{i+1}_x"][idx] * self.width)
                y = int(dataset[f"keypoint{i+1}_y"][idx] * self.height)
                x, y = self._clap(x, y)
                self.model[label][i][(x, y)] += self.landmark_weight
                if self.kernel_size > 1:
                    # kernel_size centered around (x, y)
                    for j in range(-self.kernel_size // 2, self.kernel_size // 2 + 1):
                        for k in range(-self.kernel_size // 2, self.kernel_size // 2 + 1):
                            x1 = x + j
                            y1 = y + k
                            x1, y1 = self._clap(x1, y1)
                            if x1 == x and y1 == y:
                                continue
                            if self.kernel_decay_method == "distance":
                                self.model[label][i][(x1, y1)] += self.landmark_weight / self._dist((x, y), (x1, y1))
                            else:
                                self.model[label][i][(x1, y1)] += self.landmark_weight
        
        for label in unique_labels:
            for i in range(21):
                total = 0
                for j in range(self.width):
                    for k in range(self.height):
                        total += self.model[label][i][(j, k)]
                for j in range(self.width):
                    for k in range(self.height):
                        self.model[label][i][(j, k)] /= total
        
        for label in unique_labels:
            self.class_probabilities[label] = 0
        for idx in dataset.index:
            label = dataset["label"][idx]
            self.class_probabilities[label] += 1
        for label in unique_labels:
            self.class_probabilities[label] /= len(dataset)
        return

    def _preprocess_landmarks(self, dataset):
        if self.prob_type == "collective":
            self._preprocess_landmarks_collective(dataset)
        elif self.prob_type == "individual":
            self._preprocess_landmarks_individual(dataset)


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
    

    def _classify_landmarks_collective(self, dataset):
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
    

    def _classify_landmarks_individual(self, dataset):
        unique_labels = dataset['label'].unique()

        for idx in dataset.index:
            ps = {}
            for label in unique_labels:
                p = 1
                for i in range(21):
                    x = int(dataset[f"keypoint{i+1}_x"][idx] * self.width)
                    y = int(dataset[f"keypoint{i+1}_y"][idx] * self.height)
                    x, y = self._clap(x, y)
                    p += math.log(self.model[label][i][(x, y)])
                p *= self.class_probabilities[label]
                ps[label] = p
            max_label = max(ps, key=ps.get)
            dataset.at[idx, "predicted_label"] = max_label
        return dataset
    

    def _classify_landmarks(self, dataset):
        if self.prob_type == "collective":
            preds = self._classify_landmarks_collective(dataset)
        elif self.prob_type == "individual":
            preds = self._classify_landmarks_individual(dataset)
        return preds


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
    
