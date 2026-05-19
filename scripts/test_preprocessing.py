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
clip_limits = [1.0, 2.0, 4.0]

normalization_results = []
local_histeq_results = []
clipped_histeq_results = []


for ws in window_sizes:

    norm = local_normalization(
        img,
        window_size=ws,
    )

    local_eq = local_histogram_equalization(
        img,
        window_size=ws,
        stride=1,
        clip_histogram=False,
    )

    normalization_results.append((ws, norm))
    local_histeq_results.append((ws, local_eq))

    for clip_limit in clip_limits:

        clipped_eq = local_histogram_equalization(
            img,
            window_size=ws,
            stride=1,
            clip_histogram=True,
            clip_limit=clip_limit,
        )

        clipped_histeq_results.append((ws, clip_limit, clipped_eq))


global_eq = cv2.equalizeHist(img)


# =========================
# Figure 1: main comparison
# =========================

num_cols = len(window_sizes) + 1

plt.figure(figsize=(18, 10))


# Row 1 -> Local normalization
plt.subplot(3, num_cols, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

for i, (ws, norm) in enumerate(normalization_results, start=2):
    plt.subplot(3, num_cols, i)
    plt.imshow(norm, cmap="gray")
    plt.title(f"Norm {ws}")
    plt.axis("off")


# Row 2 -> Local histogram equalization without clipping
plt.subplot(3, num_cols, num_cols + 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

for i, (ws, eq) in enumerate(local_histeq_results, start=2):
    plt.subplot(3, num_cols, num_cols + i)
    plt.imshow(eq, cmap="gray")
    plt.title(f"Local HistEq {ws}")
    plt.axis("off")


# Row 3 -> Global histogram equalization
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


# =========================
# Figure 2: clipped local histogram equalization
# =========================

num_rows = len(window_sizes)
num_cols = len(clip_limits) + 1

plt.figure(figsize=(18, 12))

plot_idx = 1

for ws in window_sizes:

    plt.subplot(num_rows, num_cols, plot_idx)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original | ws={ws}")
    plt.axis("off")
    plot_idx += 1

    for clip_limit in clip_limits:

        clipped_eq = local_histogram_equalization(
            img,
            window_size=ws,
            stride=1,
            clip_histogram=True,
            clip_limit=clip_limit,
        )

        plt.subplot(num_rows, num_cols, plot_idx)
        plt.imshow(clipped_eq, cmap="gray")
        plt.title(f"Clip {clip_limit}")
        plt.axis("off")
        plot_idx += 1


plt.tight_layout()
plt.show()