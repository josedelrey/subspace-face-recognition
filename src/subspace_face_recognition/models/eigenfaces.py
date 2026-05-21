import numpy as np


class EigenFaces:
    def __init__(self, eps: float = 1e-10):
        self.eps = eps

        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.image_shape_ = None

    def fit(self, X: np.ndarray, y=None, image_shape=None) -> None:
        X = X.astype(np.float64)

        n_samples, n_features = X.shape

        self.image_shape_ = image_shape
        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_

        A = X_centered.T

        C_prime = (A.T @ A) / n_features

        eigenvalues_prime, eigenvectors_prime = np.linalg.eigh(C_prime)

        order = np.argsort(eigenvalues_prime)[::-1]

        eigenvalues_prime = eigenvalues_prime[order]
        eigenvectors_prime = eigenvectors_prime[:, order]

        valid = eigenvalues_prime > self.eps

        eigenvalues_prime = eigenvalues_prime[valid]
        eigenvectors_prime = eigenvectors_prime[:, valid]

        eigenvectors = A @ eigenvectors_prime

        norms = np.linalg.norm(eigenvectors, axis=0)

        valid = norms > self.eps

        eigenvectors = eigenvectors[:, valid]
        eigenvalues_prime = eigenvalues_prime[valid]
        norms = norms[valid]

        eigenvectors = eigenvectors / norms

        eigenvalues = (n_features / n_samples) * eigenvalues_prime

        self.components_ = eigenvectors.T
        self.eigenvalues_ = eigenvalues

    def transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Model has not been fitted yet")

        X = X.astype(np.float64)
        X_centered = X - self.mean_

        return X_centered @ self.components_[:n_components].T

    def fit_transform(
        self,
        X: np.ndarray,
        y=None,
        image_shape=None,
        n_components: int | None = None,
    ) -> np.ndarray:
        self.fit(X, y=y, image_shape=image_shape)

        if n_components is None:
            n_components = self.components_.shape[0]

        return self.transform(X, n_components=n_components)
