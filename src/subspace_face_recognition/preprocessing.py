import numpy as np


def _to_uint8_display(
    image: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    image = (image - min_val) / (max_val - min_val + eps)
    image = image * 255.0

    return image.astype(np.uint8)


def _validate_output_type(output_type: str) -> None:
    if output_type not in {"float32", "uint8"}:
        raise ValueError("output_type must be 'float32' or 'uint8'")


def _equalize_window(
    window: np.ndarray,
    clip_histogram: bool = False,
    clip_limit: float = 2.0,
    num_bins: int = 256,
) -> np.ndarray:
    window = window.astype(np.uint8)

    hist = np.bincount(
        window.ravel(),
        minlength=num_bins,
    ).astype(np.float32)

    if clip_histogram:
        area = window.size

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


def _integral_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)

    h, w = image.shape

    integral = np.zeros(
        (h + 1, w + 1),
        dtype=np.float64,
    )

    integral[1:, 1:] = np.cumsum(
        np.cumsum(image, axis=0),
        axis=1,
    )

    return integral


def global_normalization(
    image: np.ndarray,
    eps: float = 1e-6,
    output_type: str = "float32",
) -> np.ndarray:
    """
    Global z-score normalization.

    Equivalent global version of local_normalization.

    By default, returns float32, which is suitable for PCA, Eigenfaces
    and Fisherfaces.

    Use output_type='uint8' only for visualization or saving images.
    """

    _validate_output_type(output_type)

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    std = max(std, eps)

    output = (image - mean) / std

    if output_type == "uint8":
        return _to_uint8_display(output, eps=eps)

    return output.astype(np.float32)


def global_histogram_equalization(
    image: np.ndarray,
    clip_histogram: bool = False,
    clip_limit: float = 2.0,
    output_type: str = "float32",
) -> np.ndarray:
    """
    Global histogram equalization.

    Equivalent global version of local_histogram_equalization.

    By default, returns float32, which is suitable for PCA, Eigenfaces
    and Fisherfaces.

    Use output_type='uint8' only for visualization or saving images.
    """

    _validate_output_type(output_type)

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")

    image = image.astype(np.uint8)

    output = _equalize_window(
        image,
        clip_histogram=clip_histogram,
        clip_limit=clip_limit,
    )

    if output_type == "uint8":
        return output.astype(np.uint8)

    return output.astype(np.float32)


def local_normalization(
    image: np.ndarray,
    window_size: int = 15,
    eps: float = 1e-6,
    output_type: str = "float32",
) -> np.ndarray:
    """
    Local z-score normalization.

    By default, returns float32, which is suitable for PCA, Eigenfaces
    and Fisherfaces.

    Use output_type='uint8' only for visualization or saving images.
    """

    _validate_output_type(output_type)

    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    image = image.astype(np.float32)

    h, w = image.shape
    pad = window_size // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="reflect",
    )

    padded_sq = padded ** 2

    integral = _integral_image(padded)
    integral_sq = _integral_image(padded_sq)

    area = window_size * window_size

    output = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            y0 = y
            x0 = x

            y1 = y + window_size
            x1 = x + window_size

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

            local_var = (
                window_sq_sum / area
            ) - (local_mean ** 2)

            local_var = max(local_var, 0.0)

            local_std = np.sqrt(local_var)
            local_std = max(local_std, eps)

            output[y, x] = (
                image[y, x] - local_mean
            ) / local_std

    if output_type == "uint8":
        return _to_uint8_display(output, eps=eps)

    return output.astype(np.float32)


def local_histogram_equalization(
    image: np.ndarray,
    window_size: int = 15,
    stride: int = 1,
    clip_histogram: bool = False,
    clip_limit: float = 2.0,
    output_type: str = "float32",
) -> np.ndarray:
    """
    Local histogram equalization.

    By default, returns float32, which is suitable for PCA, Eigenfaces
    and Fisherfaces.

    Use output_type='uint8' only for visualization or saving images.
    """

    _validate_output_type(output_type)

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

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="reflect",
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

    if output_type == "uint8":
        return output.astype(np.uint8)

    return output.astype(np.float32)
