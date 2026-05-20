import numpy as np

from src.metrics import accuracy_score, error_rate
from src.nearest_neighbor import NearestNeighborClassifier


def evaluate_subspace_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    component_values: list[int],
):
    results = []

    for n_components in component_values:
        Z_train = model.transform(X_train, n_components=n_components)
        Z_test = model.transform(X_test, n_components=n_components)

        classifier = NearestNeighborClassifier()
        classifier.fit(Z_train, y_train)

        y_pred = classifier.predict(Z_test)

        acc = accuracy_score(y_test, y_pred)
        err = error_rate(y_test, y_pred)

        results.append(
            {
                "n_components": n_components,
                "accuracy": acc,
                "error": err,
            }
        )

        print(
            f"d' = {n_components:3d} | "
            f"accuracy = {acc:.4f} | "
            f"error = {err:.4f}"
        )

    return results