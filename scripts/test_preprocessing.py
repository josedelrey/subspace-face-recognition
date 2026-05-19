import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing import (
    local_normalization,
    local_histogram_equalization,
)


# Change this to any image you want
image_path = Path("data/s1/1.pgm")

img = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE,
)

if img is None:
    raise ValueError(f"Could not load image: {image_path}")


window_sizes = [7, 15, 31]

normalization_results = []
equalization_results = []


for ws in window_sizes:

    norm = local_normalization(
        img,
        window_size=ws,
    )

    eq = local_histogram_equalization(
        img,
        window_size=ws,
        stride=1,
    )

    normalization_results.append((ws, norm))
    equalization_results.append((ws, eq))


# Global histogram equalization
global_eq = cv2.equalizeHist(img)


num_cols = max(
    len(normalization_results),
    len(equalization_results),
) + 1

plt.figure(figsize=(18, 10))


# =========================
# Row 1 -> Local normalization
# =========================

plt.subplot(3, num_cols, 1)

plt.imshow(img, cmap="gray")

plt.title("Original")

plt.axis("off")


for i, (ws, norm) in enumerate(
    normalization_results,
    start=2,
):

    plt.subplot(3, num_cols, i)

    plt.imshow(norm, cmap="gray")

    plt.title(f"Norm {ws}")

    plt.axis("off")


# =========================
# Row 2 -> Local histogram equalization
# =========================

plt.subplot(3, num_cols, num_cols + 1)

plt.imshow(img, cmap="gray")

plt.title("Original")

plt.axis("off")


for i, (ws, eq) in enumerate(
    equalization_results,
    start=2,
):

    plt.subplot(3, num_cols, num_cols + i)

    plt.imshow(eq, cmap="gray")

    plt.title(f"Local HistEq {ws}")

    plt.axis("off")


# =========================
# Row 3 -> Global histogram equalization
# =========================

plt.subplot(3, num_cols, 2 * num_cols + 1)

plt.imshow(img, cmap="gray")

plt.title("Original")

plt.axis("off")


plt.subplot(3, num_cols, 2 * num_cols + 2)

plt.imshow(global_eq, cmap="gray")

plt.title("Global HistEq")

plt.axis("off")


plt.tight_layout()

plt.show()