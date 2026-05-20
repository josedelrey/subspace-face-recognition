from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_error_curve(results, output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    component_values = [r["n_components"] for r in results]
    errors = [r["error"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(component_values, errors, marker="o")
    plt.xlabel("Dimensión del subespacio d'")
    plt.ylabel("Tasa de error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_mean_face(model, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_face = model.mean.reshape(model.image_shape)

    plt.figure(figsize=(4, 4))
    plt.imshow(mean_face, cmap="gray")
    plt.title("Cara promedio")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def plot_components(model, output_path: Path, n_components: int = 16):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_components = min(n_components, model.components.shape[0])

    cols = 4
    rows = int(np.ceil(n_components / cols))

    plt.figure(figsize=(10, 2.5 * rows))

    for i in range(n_components):
        component = model.components[i].reshape(model.image_shape)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(component, cmap="gray")
        plt.title(f"Componente {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()