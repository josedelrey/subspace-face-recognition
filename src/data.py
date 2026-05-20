from pathlib import Path

import cv2
import numpy as np


def load_orl_dataset(
    data_dir: Path,
    train_image_ids=range(1, 6),
    test_image_ids=range(6, 11),
    preprocessing_fn=None,
):
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    image_shape = None

    subject_dirs = sorted(
        [
            p for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("s")
        ],
        key=lambda p: int(p.name[1:]),
    )

    for subject_dir in subject_dirs:
        label = int(subject_dir.name[1:])

        for image_id in train_image_ids:
            image = _read_grayscale_image(subject_dir / f"{image_id}.pgm")

            if preprocessing_fn is not None:
                image = preprocessing_fn(image)

            if image_shape is None:
                image_shape = image.shape

            X_train.append(image.reshape(-1))
            y_train.append(label)

        for image_id in test_image_ids:
            image = _read_grayscale_image(subject_dir / f"{image_id}.pgm")

            if preprocessing_fn is not None:
                image = preprocessing_fn(image)

            X_test.append(image.reshape(-1))
            y_test.append(label)

    return (
        np.array(X_train, dtype=np.float64),
        np.array(y_train),
        np.array(X_test, dtype=np.float64),
        np.array(y_test),
        image_shape,
    )


def _read_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image