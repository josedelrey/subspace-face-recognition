import numpy as np


class DirectRepresentation:
    is_direct_representation = True

    def __init__(self, center: bool = False):
        self.center = center

        self.mean_ = None
        self.image_shape_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y=None, image_shape=None) -> None:
        X = X.astype(np.float64)

        self.image_shape_ = image_shape
        self.n_features_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)

    def transform(
        self,
        X: np.ndarray,
        n_components: int | None = None,
    ) -> np.ndarray:
        if self.n_features_ is None or self.mean_ is None:
            raise ValueError("Model has not been fitted yet")

        if n_components is not None and n_components != self.n_features_:
            raise ValueError(
                "Direct representation uses the full image dimensionality"
            )

        X = X.astype(np.float64)

        if self.center:
            return X - self.mean_

        return X

    def fit_transform(
        self,
        X: np.ndarray,
        y=None,
        image_shape=None,
        n_components: int | None = None,
    ) -> np.ndarray:
        self.fit(X, y=y, image_shape=image_shape)
        return self.transform(X, n_components=n_components)
