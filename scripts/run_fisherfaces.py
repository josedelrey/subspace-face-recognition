from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.experiments import run_experiment


DEFAULT_CONFIG = {
    "experiment": {
        "name": "fisherfaces_orl",
    },
    "data": {
        "dataset": "orl",
        "data_dir": "data",
        "train_image_ids": [1, 2, 3, 4, 5],
        "test_image_ids": [6, 7, 8, 9, 10],
    },
    "preprocessing": {
        "name": "local_normalization",
        "params": {
            "window_size": 15,
        },
    },
    "model": {
        "name": "fisherfaces",
        "params": {
            "pca_components": "auto",
            "lda_components": "auto",
            "regularization": 1e-6,
        },
    },
    "classifier": {
        "name": "nearest_neighbor",
        "params": {
            "k": 1,
            "distance": "euclidean",
        },
    },
    "projection": {
        "drop_first_components": 0,
        "normalize": False,
    },
    "components": {
        "start": 1,
        "stop": "auto",
        "step": 1,
    },
    "outputs": {
        "save_figures": True,
        "figures_dir": "outputs/figures",
        "save_error_curve": True,
        "save_mean_face": True,
        "save_components": True,
        "n_plot_components": 16,
    },
}


def main():
    rows = run_experiment(DEFAULT_CONFIG, verbose=True)
    best = min(rows, key=lambda row: float(row["error"]))

    print()
    print(f"Best d': {best['n_components']}")
    print(f"Best accuracy: {float(best['accuracy']):.4f}")
    print(f"Best error: {float(best['error']):.4f}")


if __name__ == "__main__":
    main()
