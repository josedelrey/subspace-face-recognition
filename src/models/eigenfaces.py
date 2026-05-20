import numpy as np


class EigenFaces:
    def __init__(self, eps: float = 1e-10):
        self.eps = eps

        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.image_shape = None

    def fit(self, X: np.ndarray, y=None, image_shape=None) -> None:
        X = X.astype(np.float64)

        n_samples, n_features = X.shape

        self.image_shape = image_shape
        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

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

        self.components = eigenvectors.T
        self.eigenvalues = eigenvalues

    def transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise ValueError("Model has not been fitted yet")

        X = X.astype(np.float64)
        X_centered = X - self.mean

        return X_centered @ self.components[:n_components].T

    def fit_transform(
        self,
        X: np.ndarray,
        y=None,
        image_shape=None,
        n_components: int | None = None,
    ) -> np.ndarray:
        self.fit(X, y=y, image_shape=image_shape)

        if n_components is None:
            n_components = self.components.shape[0]

        return self.transform(X, n_components=n_components)