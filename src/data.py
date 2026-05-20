from pathlib import Path

import numpy as np


def load_orl_dataset(
    data_dir: Path,
    train_image_ids=range(1, 6),
    test_image_ids=range(6, 11),
    subject_ids=None,
    preprocessing_fn=None,
):
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    image_shape = None

    selected_subject_ids = (
        None if subject_ids is None
        else {int(subject_id) for subject_id in subject_ids}
    )

    subject_dirs = sorted(
        [
            p for p in data_dir.iterdir()
            if p.is_dir() and p.name.startswith("s")
        ],
        key=lambda p: int(p.name[1:]),
    )

    for subject_dir in subject_dirs:
        label = int(subject_dir.name[1:])

        if selected_subject_ids is not None and label not in selected_subject_ids:
            continue

        for image_id in train_image_ids:
            image = read_grayscale_image(subject_dir / f"{image_id}.pgm")

            if preprocessing_fn is not None:
                image = preprocessing_fn(image)

            if image_shape is None:
                image_shape = image.shape

            X_train.append(image.reshape(-1))
            y_train.append(label)

        for image_id in test_image_ids:
            image = read_grayscale_image(subject_dir / f"{image_id}.pgm")

            if preprocessing_fn is not None:
                image = preprocessing_fn(image)

            X_test.append(image.reshape(-1))
            y_test.append(label)

    if not X_train or not X_test:
        raise ValueError("No ORL samples were loaded")

    return (
        np.array(X_train, dtype=np.float64),
        np.array(y_train),
        np.array(X_test, dtype=np.float64),
        np.array(y_test),
        image_shape,
    )


def read_grayscale_image(image_path: Path) -> np.ndarray:
    image = _read_pgm_image(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image


def _read_grayscale_image(image_path: Path) -> np.ndarray:
    return read_grayscale_image(image_path)


def _read_pgm_image(image_path: Path) -> np.ndarray:
    with image_path.open("rb") as image_file:
        magic = _read_pgm_token(image_file)

        if magic not in {b"P2", b"P5"}:
            raise ValueError(f"Unsupported PGM format in {image_path}")

        width = int(_read_pgm_token(image_file))
        height = int(_read_pgm_token(image_file))
        max_value = int(_read_pgm_token(image_file))

        if max_value <= 0:
            raise ValueError(f"Invalid PGM max value in {image_path}")

        if magic == b"P5":
            dtype = np.uint8 if max_value < 256 else ">u2"
            count = width * height
            data = np.frombuffer(
                image_file.read(count * np.dtype(dtype).itemsize),
                dtype=dtype,
            )

            if data.size != count:
                raise ValueError(f"Unexpected end of image data in {image_path}")

            image = data.reshape((height, width))
        else:
            values = [
                int(_read_pgm_token(image_file))
                for _ in range(width * height)
            ]
            image = np.array(values, dtype=np.uint16).reshape((height, width))

        if max_value > 255:
            image = (
                image.astype(np.float64) * 255.0 / max_value
            ).astype(np.uint8)

        return image.astype(np.uint8)


def _read_pgm_token(image_file) -> bytes:
    token = bytearray()

    while True:
        char = image_file.read(1)

        if char == b"":
            raise ValueError("Unexpected end of PGM header")

        if char == b"#":
            image_file.readline()
            continue

        if char.isspace():
            continue

        token.extend(char)
        break

    while True:
        char = image_file.read(1)

        if char == b"" or char.isspace():
            break

        if char == b"#":
            image_file.readline()
            break

        token.extend(char)

    return bytes(token)
