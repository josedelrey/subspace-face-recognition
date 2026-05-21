from __future__ import annotations

from typing import Any

import numpy as np


def project_subspace_features(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    projection_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    params = {} if projection_params is None else projection_params

    drop_first_components = int(params.get("drop_first_components", 0))

    if drop_first_components < 0:
        raise ValueError("drop_first_components cannot be negative")

    projected_components = n_components + drop_first_components

    if projected_components > model.components_.shape[0]:
        raise ValueError(
            "Requested projection exceeds the fitted subspace size"
        )

    Z_train = model.transform(
        X_train,
        n_components=projected_components,
    )
    Z_test = model.transform(
        X_test,
        n_components=projected_components,
    )

    if drop_first_components > 0:
        Z_train = Z_train[:, drop_first_components:]
        Z_test = Z_test[:, drop_first_components:]

    if params.get("normalize", False):
        eps = float(params.get("normalize_eps", 1e-12))
        Z_train = l2_normalize_rows(Z_train, eps=eps)
        Z_test = l2_normalize_rows(Z_test, eps=eps)

    return Z_train, Z_test


def effective_component_limit(
    model,
    projection_params: dict[str, Any] | None = None,
) -> int:
    params = {} if projection_params is None else projection_params
    drop_first_components = int(params.get("drop_first_components", 0))

    if drop_first_components < 0:
        raise ValueError("drop_first_components cannot be negative")

    limit = model.components_.shape[0] - drop_first_components

    if limit < 1:
        raise ValueError(
            "drop_first_components leaves no usable projection dimensions"
        )

    return limit


def l2_normalize_rows(
    X: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)
