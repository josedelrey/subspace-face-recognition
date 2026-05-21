from __future__ import annotations

from typing import Any

import numpy as np


def project_representation_features(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    projection_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    params = {} if projection_params is None else projection_params
    drop_first_components = _resolve_drop_first_components(params)

    if is_direct_representation(model):
        if drop_first_components != 0:
            raise ValueError(
                "drop_first_components is only valid for subspace models"
            )

        if n_components != model.n_features_:
            raise ValueError(
                "Direct representation uses the full image dimensionality"
            )

        Z_train = model.transform(X_train, n_components=n_components)
        Z_test = model.transform(X_test, n_components=n_components)
    else:
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


def default_component_values(
    model,
    component_cfg,
    max_components: int,
) -> list[int] | None:
    if not is_direct_representation(model):
        return None

    if component_cfg in (None, "auto"):
        return [max_components]

    if isinstance(component_cfg, list):
        return _validate_direct_component_values(component_cfg, max_components)

    if isinstance(component_cfg, dict):
        values = component_cfg.get("values")

        if values not in (None, "auto"):
            if isinstance(values, list):
                return _validate_direct_component_values(values, max_components)

            return _validate_direct_component_values([values], max_components)

        start = int(component_cfg.get("start", 1))
        stop = component_cfg.get("stop", "auto")
        step = int(component_cfg.get("step", 1))

        if start == 1 and stop == "auto":
            return [max_components]

        stop = max_components if stop == "auto" else int(stop)
        values = list(range(start, stop + 1, step))
        return _validate_direct_component_values(values, max_components)

    return None


def effective_component_limit(
    model,
    projection_params: dict[str, Any] | None = None,
) -> int:
    params = {} if projection_params is None else projection_params
    drop_first_components = _resolve_drop_first_components(params)

    if is_direct_representation(model):
        if drop_first_components != 0:
            raise ValueError(
                "drop_first_components is only valid for subspace models"
            )

        return model.n_features_

    limit = model.components_.shape[0] - drop_first_components

    if limit < 1:
        raise ValueError(
            "drop_first_components leaves no usable projection dimensions"
        )

    return limit


def is_direct_representation(model) -> bool:
    return bool(getattr(model, "is_direct_representation", False))


def l2_normalize_rows(
    X: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _resolve_drop_first_components(params: dict[str, Any]) -> int:
    drop_first_components = int(params.get("drop_first_components", 0))

    if drop_first_components < 0:
        raise ValueError("drop_first_components cannot be negative")

    return drop_first_components


def _validate_direct_component_values(
    values,
    max_components: int,
) -> list[int]:
    values = [int(value) for value in values]

    if values != [max_components]:
        raise ValueError(
            "Direct representation only supports the full image "
            f"dimensionality: {max_components}"
        )

    return values
