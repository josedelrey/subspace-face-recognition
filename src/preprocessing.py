import cv2
import numpy as np


def local_normalization(
    image: np.ndarray,
    window_size: int = 15,
    stride: int = 1,
    eps: float = 1e-6,
    normalize_output: bool = True,
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    image = image.astype(np.float32)

    h, w = image.shape

    pad = window_size // 2

    padded = cv2.copyMakeBorder(
        image,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_REFLECT,
    )

    acc = np.zeros_like(padded, dtype=np.float32)
    count = np.zeros_like(padded, dtype=np.float32)

    padded_h, padded_w = padded.shape

    for y in range(0, padded_h - window_size + 1, stride):
        for x in range(0, padded_w - window_size + 1, stride):

            window = padded[
                y : y + window_size,
                x : x + window_size,
            ]

            local_mean = np.mean(window)
            local_std = np.std(window)
            local_std = max(local_std, eps)

            normalized_window = (window - local_mean) / local_std

            acc[
                y : y + window_size,
                x : x + window_size,
            ] += normalized_window

            count[
                y : y + window_size,
                x : x + window_size,
            ] += 1.0

    output_padded = acc / np.maximum(count, eps)

    output = output_padded[
        pad : pad + h,
        pad : pad + w,
    ]

    if normalize_output:
        output -= output.min()
        output /= output.max() + eps
        output *= 255.0
        output = output.astype(np.uint8)

    return output