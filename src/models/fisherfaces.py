import numpy as np


class FisherFaces:
    def __init__(
        self,
        pca_components: int | str | None = "auto",
        lda_components: int | str | None = "auto",
        regularization: float = 1e-6,
        eps: float = 1e-10,
    ):
        self.pca_components = pca_components
        self.lda_components = lda_components
        self.regularization = regularization
        self.eps = eps

        self.mean = None
        self.components = None
        self.eigenvalues = None
        self.image_shape = None

        self.pca_components_ = None
        self.pca_eigenvalues = None
        self.pca_components_count = None
        self.lda_components_count = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, image_shape=None) -> None:
        if y is None:
            raise ValueError("FisherFaces requires class labels")

        X = X.astype(np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        classes = np.unique(y)

        if classes.shape[0] < 2:
            raise ValueError("FisherFaces requires at least two classes")

        n_samples, _ = X.shape
        n_classes = classes.shape[0]

        self.image_shape = image_shape
        self.classes_ = classes
        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        _, singular_values, right_vectors = np.linalg.svd(
            X_centered,
            full_matrices=False,
        )

        rank = self._matrix_rank(singular_values)
        max_pca_components = min(rank, n_samples - n_classes)

        if max_pca_components < 1:
            max_pca_components = min(rank, n_samples - 1)

        if max_pca_components < 1:
            raise ValueError("Not enough rank to compute a PCA subspace")

        pca_components_count = self._resolve_component_count(
            requested=self.pca_components,
            default=max_pca_components,
            maximum=rank,
            name="pca_components",
        )
        pca_components_count = min(pca_components_count, max_pca_components)

        self.pca_components_ = right_vectors[:pca_components_count]
        self.pca_eigenvalues = (
            singular_values[:pca_components_count] ** 2
        ) / max(n_samples - 1, 1)
        self.pca_components_count = pca_components_count

        X_pca = X_centered @ self.pca_components_.T

        within_scatter, between_scatter = self._scatter_matrices(
            X_pca,
            y,
            classes,
        )

        lda_components_available = min(n_classes - 1, pca_components_count)
        lda_components_count = self._resolve_component_count(
            requested=self.lda_components,
            default=lda_components_available,
            maximum=lda_components_available,
            name="lda_components",
        )

        lda_vectors, lda_values = self._solve_lda(
            within_scatter=within_scatter,
            between_scatter=between_scatter,
            n_components=lda_components_count,
        )

        components = lda_vectors.T @ self.pca_components_
        components = self._normalize_rows(components)

        self.components = components
        self.eigenvalues = lda_values[: components.shape[0]]
        self.lda_components_count = components.shape[0]

    def transform(
        self,
        X: np.ndarray,
        n_components: int | None = None,
    ) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise ValueError("Model has not been fitted yet")

        if n_components is None:
            n_components = self.components.shape[0]

        if n_components < 1:
            raise ValueError("n_components must be positive")

        if n_components > self.components.shape[0]:
            raise ValueError(
                "n_components exceeds the fitted Fisher subspace size"
            )

        X = X.astype(np.float64)
        X_centered = X - self.mean

        return X_centered @ self.components[:n_components].T

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        image_shape=None,
        n_components: int | None = None,
    ) -> np.ndarray:
        self.fit(X, y=y, image_shape=image_shape)
        return self.transform(X, n_components=n_components)

    def _matrix_rank(self, singular_values: np.ndarray) -> int:
        if singular_values.size == 0:
            return 0

        threshold = self.eps * max(float(singular_values[0]), 1.0)
        return int(np.sum(singular_values > threshold))

    def _resolve_component_count(
        self,
        requested,
        default: int,
        maximum: int,
        name: str,
    ) -> int:
        if requested is None or requested == "auto":
            value = default
        else:
            value = int(requested)

        if value < 1:
            raise ValueError(f"{name} must be positive")

        return min(value, maximum)

    def _scatter_matrices(
        self,
        X_pca: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_features = X_pca.shape[1]
        overall_mean = np.mean(X_pca, axis=0)

        within_scatter = np.zeros((n_features, n_features), dtype=np.float64)
        between_scatter = np.zeros((n_features, n_features), dtype=np.float64)

        for class_label in classes:
            class_samples = X_pca[y == class_label]
            class_mean = np.mean(class_samples, axis=0)

            centered = class_samples - class_mean
            within_scatter += centered.T @ centered

            mean_delta = (class_mean - overall_mean).reshape(-1, 1)
            between_scatter += class_samples.shape[0] * (mean_delta @ mean_delta.T)

        return within_scatter, between_scatter

    def _solve_lda(
        self,
        within_scatter: np.ndarray,
        between_scatter: np.ndarray,
        n_components: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_features = within_scatter.shape[0]
        scale = np.trace(within_scatter) / max(n_features, 1)

        if not np.isfinite(scale) or scale <= self.eps:
            scale = 1.0

        within_scatter = within_scatter + (
            self.regularization * scale * np.eye(n_features)
        )

        sw_values, sw_vectors = np.linalg.eigh(within_scatter)
        max_sw_value = max(float(np.max(np.abs(sw_values))), 1.0)
        valid = sw_values > (self.eps * max_sw_value)

        if not np.any(valid):
            raise ValueError("Within-class scatter matrix is numerically singular")

        whitening = sw_vectors[:, valid] @ np.diag(1.0 / np.sqrt(sw_values[valid]))
        lda_matrix = whitening.T @ between_scatter @ whitening
        lda_matrix = 0.5 * (lda_matrix + lda_matrix.T)

        lda_values, lda_vectors = np.linalg.eigh(lda_matrix)
        order = np.argsort(lda_values)[::-1]

        lda_values = lda_values[order][:n_components]
        lda_vectors = whitening @ lda_vectors[:, order[:n_components]]

        return lda_vectors, lda_values

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1)
        valid = norms > self.eps

        if not np.all(valid):
            matrix = matrix[valid]
            norms = norms[valid]

        if matrix.shape[0] == 0:
            raise ValueError("No valid Fisherfaces were found")

        return matrix / norms[:, None]
