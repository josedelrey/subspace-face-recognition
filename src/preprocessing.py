import cv2
import numpy as np


def local_normalization(
    image: np.ndarray,
    window_size: int = 15,
    eps: float = 1e-6,
    normalize_output: bool = True,
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

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

    padded_sq = padded ** 2

    # Integral images
    integral = cv2.integral(padded)
    integral_sq = cv2.integral(padded_sq)

    area = window_size * window_size

    output = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            y0 = y
            x0 = x
            y1 = y + window_size
            x1 = x + window_size

            # Sum over local window using integral image
            window_sum = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )

            window_sq_sum = (
                integral_sq[y1, x1]
                - integral_sq[y0, x1]
                - integral_sq[y1, x0]
                + integral_sq[y0, x0]
            )

            local_mean = window_sum / area

            local_var = (window_sq_sum / area) - (local_mean ** 2)
            local_var = max(local_var, 0.0)

            local_std = np.sqrt(local_var)
            local_std = max(local_std, eps)

            # Normalize center pixel using its local neighborhood
            output[y, x] = (image[y, x] - local_mean) / local_std

    if normalize_output:
        output -= output.min()
        output /= output.max() + eps
        output *= 255.0
        output = output.astype(np.uint8)

    return output


def local_histogram_equalization(
    image: np.ndarray,
    window_size: int = 15,
    stride: int = 1,
    normalize_output: bool = True,
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    image = image.astype(np.uint8)

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

            equalized_window = cv2.equalizeHist(window)

            acc[
                y : y + window_size,
                x : x + window_size,
            ] += equalized_window.astype(np.float32)

            count[
                y : y + window_size,
                x : x + window_size,
            ] += 1.0

    output_padded = acc / np.maximum(count, 1.0)

    output = output_padded[
        pad : pad + h,
        pad : pad + w,
    ]

    if normalize_output:
        output -= output.min()
        output /= output.max() + 1e-6
        output *= 255.0

    return output.astype(np.uint8)