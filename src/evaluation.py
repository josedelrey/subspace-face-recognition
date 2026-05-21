import numpy as np

from src.features import project_subspace_features
from src.metrics import accuracy_score, error_rate
from src.knn import KNNClassifier


def evaluate_subspace_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    component_values: list[int],
    projection_params: dict | None = None,
    classifier_params: dict | None = None,
):
    results = []
    classifier_params = {} if classifier_params is None else classifier_params

    for n_components in component_values:
        Z_train, Z_test = project_subspace_features(
            model=model,
            X_train=X_train,
            X_test=X_test,
            n_components=n_components,
            projection_params=projection_params,
        )

        classifier = KNNClassifier(**classifier_params)
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

    return results
