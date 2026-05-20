import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import read_grayscale_image
from src.preprocessing import (
    local_normalization,
    local_histogram_equalization,
)


image_path = Path("data/s1/1.pgm")
img = read_grayscale_image(image_path)


window_sizes = [7, 15, 31]
clip_limits = [2.0, 4.0, 6.0]

normalization_results = []
local_histeq_results = []
clipped_histeq_results = []


for ws in window_sizes:
    normalization_results.append(
        (
            ws,
            local_normalization(img, window_size=ws),
        )
    )

    local_histeq_results.append(
        (
            ws,
            local_histogram_equalization(
                img,
                window_size=ws,
                stride=1,
                clip_histogram=False,
            ),
        )
    )

    for clip_limit in clip_limits:
        clipped_histeq_results.append(
            (
                ws,
                clip_limit,
                local_histogram_equalization(
                    img,
                    window_size=ws,
                    stride=1,
                    clip_histogram=True,
                    clip_limit=clip_limit,
                ),
            )
        )


num_rows = 2 + len(window_sizes)
num_cols = max(len(window_sizes) + 1, len(clip_limits) + 1)

plt.figure(figsize=(20, 16))


# Row 1: Local normalization
plt.subplot(num_rows, num_cols, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

for col, (ws, norm) in enumerate(normalization_results, start=2):
    plt.subplot(num_rows, num_cols, col)
    plt.imshow(norm, cmap="gray")
    plt.title(f"Norm {ws}")
    plt.axis("off")


# Row 2: Local histogram equalization without clipping
row_start = num_cols + 1

plt.subplot(num_rows, num_cols, row_start)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

for col, (ws, eq) in enumerate(local_histeq_results, start=1):
    plt.subplot(num_rows, num_cols, row_start + col)
    plt.imshow(eq, cmap="gray")
    plt.title(f"Local HistEq {ws}")
    plt.axis("off")


# Rows 3 onwards: clipped local histogram equalization
for row_idx, ws in enumerate(window_sizes):
    row_start = (2 + row_idx) * num_cols + 1

    plt.subplot(num_rows, num_cols, row_start)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original | ws={ws}")
    plt.axis("off")

    for col_idx, clip_limit in enumerate(clip_limits, start=1):
        clipped_eq = next(
            result
            for result_ws, result_clip, result in clipped_histeq_results
            if result_ws == ws and result_clip == clip_limit
        )

        plt.subplot(num_rows, num_cols, row_start + col_idx)
        plt.imshow(clipped_eq, cmap="gray")
        plt.title(f"Clip {clip_limit}")
        plt.axis("off")


plt.tight_layout()
plt.show()
