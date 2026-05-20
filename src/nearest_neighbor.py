import numpy as np


class NearestNeighborClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier has not been fitted yet")

        predictions = []

        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_index = np.argmin(distances)
            predictions.append(self.y_train[nearest_index])

        return np.array(predictions)