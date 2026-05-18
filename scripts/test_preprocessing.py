import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing import local_normalization


# Change this to any image you want
image_path = Path("data/test.jpg")

img = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    raise ValueError(f"Could not load image: {image_path}")


window_sizes = [7, 15, 31]

results = []

for ws in window_sizes:

    norm = local_normalization(
        img,
        window_size=ws,
        stride=1,
    )

    results.append((ws, norm))


plt.figure(figsize=(16, 5))

# Original image
plt.subplot(1, len(results) + 1, 1)

plt.imshow(img, cmap="gray")

plt.title("Original")

plt.axis("off")


# Normalized versions
for i, (ws, norm) in enumerate(results, start=2):

    plt.subplot(1, len(results) + 1, i)

    plt.imshow(norm, cmap="gray")

    plt.title(f"Window {ws}")

    plt.axis("off")


plt.tight_layout()

plt.show()