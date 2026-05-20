from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_orl_dataset
from src.models.eigenfaces import EigenFaces
from src.preprocessing import local_normalization
from src.train import evaluate_subspace_model
from src.visualization import (
    plot_components,
    plot_error_curve,
    plot_mean_face,
)


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


USE_PREPROCESSING = True


def preprocess(image):
    return local_normalization(
        image,
        window_size=15,
    )


def main():
    preprocessing_fn = preprocess if USE_PREPROCESSING else None

    X_train, y_train, X_test, y_test, image_shape = load_orl_dataset(
        DATA_DIR,
        train_image_ids=range(1, 6),
        test_image_ids=range(6, 11),
        preprocessing_fn=preprocessing_fn,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Image shape: {image_shape}")

    model = EigenFaces()
    model.fit(
        X_train,
        y=y_train,
        image_shape=image_shape,
    )

    max_components = min(
        model.components.shape[0],
        X_train.shape[0] - 1,
    )

    component_values = list(range(1, max_components + 1))

    results = evaluate_subspace_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        component_values=component_values,
    )

    best_result = min(results, key=lambda r: r["error"])

    print()
    print(f"Best d': {best_result['n_components']}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    print(f"Best error: {best_result['error']:.4f}")

    plot_error_curve(
        results,
        output_path=OUTPUT_DIR / "figures" / "eigenfaces_error_curve.png",
        title="Curva de error - Eigenfaces",
    )

    plot_mean_face(
        model,
        output_path=OUTPUT_DIR / "figures" / "eigenfaces_mean_face.png",
    )

    plot_components(
        model,
        output_path=OUTPUT_DIR / "figures" / "eigenfaces_components.png",
        n_components=16,
    )


if __name__ == "__main__":
    main()