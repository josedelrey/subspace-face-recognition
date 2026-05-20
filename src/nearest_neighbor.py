import numpy as np


class NearestNeighborClassifier:
    def __init__(
        self,
        k: int = 1,
        distance: str = "euclidean",
        eps: float = 1e-12,
    ):
        if k < 1:
            raise ValueError("k must be positive")

        if distance not in {"euclidean", "cosine"}:
            raise ValueError("distance must be 'euclidean' or 'cosine'")

        self.k = k
        self.distance = distance
        self.eps = eps
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier has not been fitted yet")

        if self.k > self.X_train.shape[0]:
            raise ValueError("k cannot be larger than the training set")

        distances = self._pairwise_distances(X)
        predictions = []

        for sample_distances in distances:
            nearest_indices = np.argsort(sample_distances)[: self.k]
            nearest_labels = self.y_train[nearest_indices]
            nearest_distances = sample_distances[nearest_indices]
            predictions.append(
                self._majority_vote(nearest_labels, nearest_distances)
            )

        return np.array(predictions)

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        if self.distance == "euclidean":
            diff = X[:, None, :] - self.X_train[None, :, :]
            return np.linalg.norm(diff, axis=2)

        X_norm = self._l2_normalize(X)
        train_norm = self._l2_normalize(self.X_train)
        similarities = X_norm @ train_norm.T
        return 1.0 - similarities

    def _l2_normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, self.eps)

    def _majority_vote(
        self,
        labels: np.ndarray,
        distances: np.ndarray,
    ):
        unique_labels, counts = np.unique(labels, return_counts=True)
        best_count = np.max(counts)
        candidates = unique_labels[counts == best_count]

        if candidates.shape[0] == 1:
            return candidates[0]

        best_label = None
        best_distance = np.inf

        for label in candidates:
            label_distance = np.min(distances[labels == label])

            if label_distance < best_distance:
                best_distance = label_distance
                best_label = label

        return best_label
