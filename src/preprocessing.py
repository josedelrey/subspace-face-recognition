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

    mean = cv2.blur(
        image,
        (window_size, window_size),
        borderType=cv2.BORDER_REFLECT,
    )

    mean_sq = cv2.blur(
        image ** 2,
        (window_size, window_size),
        borderType=cv2.BORDER_REFLECT,
    )

    var = mean_sq - mean ** 2
    var = np.maximum(var, 0.0)

    std = np.sqrt(var)

    output = (image - mean) / (std + eps)

    if normalize_output:
        output -= output.min()
        output /= output.max() + eps
        output *= 255.0
        output = output.astype(np.uint8)

    return output


def _equalize_window(
    window: np.ndarray,
    clip_histogram: bool = False,
    clip_limit: float = 2.0,
    num_bins: int = 256,
) -> np.ndarray:

    if not clip_histogram:
        return cv2.equalizeHist(window)

    area = window.size

    hist = np.bincount(
        window.ravel(),
        minlength=num_bins,
    ).astype(np.float32)

    max_bin_count = max(
        1,
        int(clip_limit * area / num_bins),
    )

    excess = np.maximum(hist - max_bin_count, 0.0)
    excess_total = np.sum(excess)

    hist = np.minimum(hist, max_bin_count)

    hist += excess_total / num_bins

    cdf = np.cumsum(hist)

    if cdf[-1] <= 0:
        return window.copy()

    cdf /= cdf[-1]

    lut = np.floor(255 * cdf).astype(np.uint8)

    return lut[window]


def local_histogram_equalization(
    image: np.ndarray,
    window_size: int = 15,
    stride: int = 1,
    clip_histogram: bool = False,
    clip_limit: float = 2.0,
    normalize_output: bool = True,
) -> np.ndarray:

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")

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

            equalized_window = _equalize_window(
                window,
                clip_histogram=clip_histogram,
                clip_limit=clip_limit,
            )

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