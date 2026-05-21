from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.data import load_orl_dataset
from src.features import effective_component_limit
from src.models.eigenfaces import EigenFaces
from src.models.fisherfaces import FisherFaces
from src.preprocessing import (
    global_histogram_equalization,
    global_normalization,
    local_histogram_equalization,
    local_normalization,
)
from src.evaluation import evaluate_subspace_model
from src.visualization import plot_components, plot_error_curve, plot_mean_face


RESULT_FIELDNAMES = [
    "run_id",
    "config_file",
    "experiment_name",
    "dataset",
    "data_dir",
    "model",
    "model_params",
    "classifier",
    "classifier_params",
    "classifier_k",
    "distance",
    "projection_params",
    "drop_first_components",
    "feature_normalize",
    "preprocessing",
    "preprocessing_params",
    "subject_ids",
    "train_image_ids",
    "test_image_ids",
    "image_height",
    "image_width",
    "n_train",
    "n_test",
    "n_classes",
    "n_features",
    "pca_components",
    "max_components",
    "n_components",
    "accuracy",
    "error",
    "elapsed_seconds",
]


PREPROCESSING_FUNCTIONS = {
    "global_normalization": global_normalization,
    "global_histogram_equalization": global_histogram_equalization,
    "local_normalization": local_normalization,
    "local_histogram_equalization": local_histogram_equalization,
}


@dataclass(frozen=True)
class ExperimentPlanResult:
    rows: list[dict[str, Any]]
    results_csv: Path


def load_experiment_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read experiment plans. "
            "Install project dependencies with: pip install -e ."
        ) from exc

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")

    return config


def run_experiment_plan(
    plan_path: Path,
    results_csv: Path | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> ExperimentPlanResult:
    plan = load_experiment_config(plan_path)
    defaults = plan.get("defaults", {})
    experiments = plan.get("experiments")

    if not isinstance(defaults, dict):
        raise ValueError("Experiment plan 'defaults' must be a mapping")

    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Experiment plan must define a non-empty 'experiments' list")

    output_csv = results_csv or _resolve_plan_output_csv(plan)

    if overwrite and output_csv.exists():
        output_csv.unlink()

    all_rows = []

    for index, experiment_config in enumerate(experiments, start=1):
        if not isinstance(experiment_config, dict):
            raise ValueError(f"Experiment #{index} must be a mapping")

        config = _deep_merge(defaults, experiment_config)
        config = _ensure_experiment_name(
            config=config,
            fallback_name=f"{plan_path.stem}_{index:03d}",
        )

        if verbose:
            experiment_name = config.get("experiment", {}).get("name")
            print(f"\n=== Running {experiment_name} ({index}/{len(experiments)}) ===")

        rows = run_experiment(
            config=config,
            config_path=plan_path,
            verbose=verbose,
        )

        append_results(rows, output_csv)
        all_rows.extend(rows)

        if verbose:
            best = min(rows, key=lambda row: float(row["error"]))
            print(
                "Best "
                f"d'={best['n_components']} | "
                f"accuracy={float(best['accuracy']):.4f} | "
                f"error={float(best['error']):.4f}"
            )
            print(f"Appended {len(rows)} rows to {output_csv}")

    return ExperimentPlanResult(rows=all_rows, results_csv=output_csv)


def run_experiment(
    config: dict[str, Any],
    config_path: Path | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    start_time = time.perf_counter()

    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    classifier_cfg = config.get("classifier", {})
    projection_cfg = config.get("projection", {})
    outputs_cfg = config.get("outputs", {})

    experiment_name = experiment_cfg.get(
        "name",
        config_path.stem if config_path is not None else "experiment",
    )
    run_id = experiment_cfg.get("run_id") or _safe_slug(experiment_name)

    dataset = data_cfg.get("dataset", "orl")

    if dataset != "orl":
        raise ValueError("Only the ORL dataset is supported")

    data_dir = Path(data_cfg.get("data_dir", "data"))
    train_image_ids = _resolve_int_sequence(
        data_cfg.get("train_image_ids"),
        default=range(1, 6),
    )
    test_image_ids = _resolve_int_sequence(
        data_cfg.get("test_image_ids"),
        default=range(6, 11),
    )
    subject_ids = _resolve_optional_int_sequence(data_cfg.get("subject_ids"))

    preprocessing_name, preprocessing_params, preprocessing_fn = (
        _build_preprocessing(config.get("preprocessing"))
    )

    X_train, y_train, X_test, y_test, image_shape = load_orl_dataset(
        data_dir=data_dir,
        train_image_ids=train_image_ids,
        test_image_ids=test_image_ids,
        subject_ids=subject_ids,
        preprocessing_fn=preprocessing_fn,
    )

    model_name, model_params, model = _build_model(model_cfg)
    classifier_name, classifier_params = _resolve_classifier_config(classifier_cfg)
    projection_params = _resolve_projection_config(projection_cfg)

    if verbose:
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape:  {X_test.shape}")
        print(f"Image shape: {image_shape}")
        print(f"Model: {model_name}")
        print(f"Preprocessing: {preprocessing_name}")
        print(f"Classifier: {classifier_name} {classifier_params}")
        print(f"Projection: {projection_params}")

    model.fit(X_train, y=y_train, image_shape=image_shape)

    max_components = effective_component_limit(
        model,
        projection_params=projection_params,
    )
    component_values = _resolve_component_values(
        config.get("components"),
        max_components=max_components,
    )

    results = evaluate_subspace_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        component_values=component_values,
        projection_params=projection_params,
        classifier_params=classifier_params,
    )

    elapsed_seconds = time.perf_counter() - start_time
    rows = _build_result_rows(
        results=results,
        run_id=run_id,
        config_path=config_path,
        experiment_name=experiment_name,
        dataset=dataset,
        data_dir=data_dir,
        model_name=model_name,
        model_params=model_params,
        classifier_name=classifier_name,
        classifier_params=classifier_params,
        projection_params=projection_params,
        preprocessing_name=preprocessing_name,
        preprocessing_params=preprocessing_params,
        subject_ids=subject_ids,
        train_image_ids=train_image_ids,
        test_image_ids=test_image_ids,
        image_shape=image_shape,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        model=model,
        elapsed_seconds=elapsed_seconds,
    )

    _save_figures_if_requested(
        outputs_cfg=outputs_cfg,
        experiment_name=experiment_name,
        model=model,
        results=results,
    )

    return rows


def append_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    if file_exists:
        _validate_csv_header(output_path)

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def _validate_csv_header(output_path: Path) -> None:
    with output_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)

    if header != RESULT_FIELDNAMES:
        raise ValueError(
            f"CSV schema mismatch in {output_path}. "
            "Use --overwrite or choose a new --results-csv path."
        )


def _build_preprocessing(
    preprocessing_cfg,
) -> tuple[str, dict[str, Any], Callable[[np.ndarray], np.ndarray] | None]:
    if preprocessing_cfg in (None, False, "none"):
        return "none", {}, None

    if isinstance(preprocessing_cfg, str):
        name = preprocessing_cfg
        params = {}
    else:
        name = preprocessing_cfg.get("name", "none")
        params = preprocessing_cfg.get("params", {})

    if name in (None, "none"):
        return "none", {}, None

    if name not in PREPROCESSING_FUNCTIONS:
        raise ValueError(f"Unknown preprocessing function: {name}")

    preprocessing_function = PREPROCESSING_FUNCTIONS[name]

    def preprocess(image: np.ndarray) -> np.ndarray:
        return preprocessing_function(image, **params)

    return name, params, preprocess


def _build_model(model_cfg) -> tuple[str, dict[str, Any], Any]:
    if isinstance(model_cfg, str):
        name = model_cfg
        params = {}
    else:
        name = model_cfg.get("name", "eigenfaces")
        params = model_cfg.get("params", {})

    if name == "eigenfaces":
        return name, params, EigenFaces(**params)

    if name == "fisherfaces":
        return name, params, FisherFaces(**params)

    raise ValueError(f"Unknown model: {name}")


def _resolve_classifier_config(classifier_cfg) -> tuple[str, dict[str, Any]]:
    if classifier_cfg in (None, {}, "nearest_neighbor"):
        return "nearest_neighbor", {}

    if isinstance(classifier_cfg, str):
        name = classifier_cfg
        params = {}
    else:
        name = classifier_cfg.get("name", "nearest_neighbor")
        params = classifier_cfg.get("params", {})

    if name != "nearest_neighbor":
        raise ValueError("Only nearest_neighbor classifier is supported")

    return name, params


def _resolve_projection_config(projection_cfg) -> dict[str, Any]:
    if projection_cfg in (None, False):
        projection_cfg = {}

    if not isinstance(projection_cfg, dict):
        raise ValueError("projection must be a mapping")

    return {
        "drop_first_components": int(
            projection_cfg.get("drop_first_components", 0)
        ),
        "normalize": bool(projection_cfg.get("normalize", False)),
        "normalize_eps": float(projection_cfg.get("normalize_eps", 1e-12)),
    }


def _resolve_component_values(
    component_cfg,
    max_components: int,
) -> list[int]:
    if component_cfg in (None, "auto"):
        return list(range(1, max_components + 1))

    if isinstance(component_cfg, list):
        values = [int(value) for value in component_cfg]
        return _validate_component_values(values, max_components)

    if isinstance(component_cfg, dict):
        values = component_cfg.get("values")

        if values not in (None, "auto"):
            return _resolve_component_values(values, max_components)

        start = int(component_cfg.get("start", 1))
        stop = component_cfg.get("stop", "auto")
        step = int(component_cfg.get("step", 1))
        stop = max_components if stop == "auto" else int(stop)

        values = list(range(start, stop + 1, step))
        return _validate_component_values(values, max_components)

    raise ValueError("components must be 'auto', a list, or a mapping")


def _validate_component_values(
    values: list[int],
    max_components: int,
) -> list[int]:
    if not values:
        raise ValueError("At least one component value is required")

    invalid_values = [
        value for value in values
        if value < 1 or value > max_components
    ]

    if invalid_values:
        raise ValueError(
            "Component values out of range "
            f"1..{max_components}: {invalid_values}"
        )

    return values


def _resolve_int_sequence(value, default) -> list[int]:
    if value is None:
        return [int(item) for item in default]

    if isinstance(value, list):
        return [int(item) for item in value]

    if isinstance(value, dict):
        start = int(value.get("start", 1))
        stop = int(value["stop"])
        step = int(value.get("step", 1))
        inclusive = bool(value.get("inclusive", True))
        end = stop + 1 if inclusive else stop
        return list(range(start, end, step))

    raise ValueError("Expected a list or range mapping")


def _resolve_optional_int_sequence(value) -> list[int] | None:
    if value is None:
        return None

    return _resolve_int_sequence(value, default=[])


def _build_result_rows(
    results: list[dict[str, Any]],
    run_id: str,
    config_path: Path | None,
    experiment_name: str,
    dataset: str,
    data_dir: Path,
    model_name: str,
    model_params: dict[str, Any],
    classifier_name: str,
    classifier_params: dict[str, Any],
    projection_params: dict[str, Any],
    preprocessing_name: str,
    preprocessing_params: dict[str, Any],
    subject_ids: list[int] | None,
    train_image_ids: list[int],
    test_image_ids: list[int],
    image_shape,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    model,
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    pca_components = getattr(model, "pca_components_count_", "")
    image_height, image_width = image_shape
    classifier_k = classifier_params.get("k", 1)
    distance = classifier_params.get("distance", "euclidean")
    max_components = effective_component_limit(
        model,
        projection_params=projection_params,
    )
    rows = []

    for result in results:
        rows.append(
            {
                "run_id": run_id,
                "config_file": "" if config_path is None else str(config_path),
                "experiment_name": experiment_name,
                "dataset": dataset,
                "data_dir": str(data_dir),
                "model": model_name,
                "model_params": _to_json(model_params),
                "classifier": classifier_name,
                "classifier_params": _to_json(classifier_params),
                "classifier_k": classifier_k,
                "distance": distance,
                "projection_params": _to_json(projection_params),
                "drop_first_components": projection_params[
                    "drop_first_components"
                ],
                "feature_normalize": projection_params["normalize"],
                "preprocessing": preprocessing_name,
                "preprocessing_params": _to_json(preprocessing_params),
                "subject_ids": _to_json(subject_ids),
                "train_image_ids": _to_json(train_image_ids),
                "test_image_ids": _to_json(test_image_ids),
                "image_height": image_height,
                "image_width": image_width,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "n_classes": np.unique(y_train).shape[0],
                "n_features": X_train.shape[1],
                "pca_components": pca_components,
                "max_components": max_components,
                "n_components": result["n_components"],
                "accuracy": result["accuracy"],
                "error": result["error"],
                "elapsed_seconds": f"{elapsed_seconds:.6f}",
            }
        )

    return rows


def _save_figures_if_requested(
    outputs_cfg: dict[str, Any],
    experiment_name: str,
    model,
    results: list[dict[str, Any]],
) -> None:
    if not outputs_cfg.get("save_figures", False):
        return

    figures_dir = Path(outputs_cfg.get("figures_dir", "outputs/figures"))
    prefix = _safe_slug(experiment_name)

    if outputs_cfg.get("save_error_curve", True):
        plot_error_curve(
            results,
            output_path=figures_dir / f"{prefix}_error_curve.png",
            title=f"Curva de error - {experiment_name}",
            show=False,
        )

    if outputs_cfg.get("save_mean_face", True):
        plot_mean_face(
            model,
            output_path=figures_dir / f"{prefix}_mean_face.png",
            show=False,
        )

    if outputs_cfg.get("save_components", True):
        plot_components(
            model,
            output_path=figures_dir / f"{prefix}_components.png",
            n_components=int(outputs_cfg.get("n_plot_components", 16)),
            show=False,
        )


def _resolve_plan_output_csv(plan: dict[str, Any]) -> Path:
    outputs_cfg = plan.get("outputs")

    if outputs_cfg is None:
        outputs_cfg = plan.get("defaults", {}).get("outputs", {})

    return Path(outputs_cfg.get("results_csv", "results/experiments.csv"))


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        base_value = merged.get(key)

        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value

    return merged


def _ensure_experiment_name(
    config: dict[str, Any],
    fallback_name: str,
) -> dict[str, Any]:
    experiment_cfg = dict(config.get("experiment", {}))

    if not experiment_cfg.get("name"):
        experiment_cfg["name"] = fallback_name

    config = dict(config)
    config["experiment"] = experiment_cfg
    return config


def _to_json(value) -> str:
    return json.dumps(value, sort_keys=True)


def _safe_slug(value: str) -> str:
    allowed = []

    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", " "}:
            allowed.append("_")

    slug = "".join(allowed).strip("_")
    return slug or "experiment"
