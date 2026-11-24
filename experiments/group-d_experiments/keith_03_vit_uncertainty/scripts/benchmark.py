"""Measure how fast this computer runs the two transformers and estimate the run time of every step.

Run from the repository root (takes 1-2 minutes, downloads about 180 MB of weights once):

    uv run python experiments/group-d_experiments/keith_03_vit_uncertainty/scripts/benchmark.py

The output is a small table: images per second for full and partially hidden images, and the
estimated minutes for the feature caches, one explanation (budget 1000), and the 37-image cohort.
"""

from __future__ import annotations

from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vit_uncertainty.backbones import FrozenViT, load_backbones


def images_per_second(backbone: FrozenViT, images: np.ndarray, visible: np.ndarray | None) -> float:
    """Median throughput over three timed runs after one warm-up run."""
    backbone.features(images[:4], None if visible is None else visible[:4])
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        backbone.features(images, visible)
        timings.append(time.perf_counter() - start)
    return len(images) / float(np.median(timings))


def main() -> None:
    """Print speeds and run-time estimates."""
    threads = torch.get_num_threads()
    print(f"Python {platform.python_version()} | torch {torch.__version__} | CPU-Threads: {threads}")
    print(f"CUDA/ROCm verfügbar: {torch.cuda.is_available()} (das Paket rechnet auf der CPU)")
    backbones = load_backbones()
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(32, 224, 224, 3)).astype(np.uint8)
    speed = {}
    for name, backbone in backbones.items():
        half = rng.random((32, backbone.n_patches)) < 0.5
        speed[name] = (images_per_second(backbone, images, None), images_per_second(backbone, images, half))
        full, half_hidden = speed[name]
        print(f"{backbone.label:12s}: {full:6.1f} Bilder/s voll, {half_hidden:6.1f} Bilder/s halb ausgeblendet")
    # One coalition costs one pass through each backbone.
    seconds_per_coalition = sum(1 / masked for _, masked in speed.values())
    seconds_per_image = sum(1 / full for full, _ in speed.values())
    estimates = {
        "Merkmale aller 7349 Bilder (einmalig)": 7349 * seconds_per_image,
        "Maskierte Trainingsdaten (2944 x 8 + 736 x 4 Koalitionen)": (2944 * 8 + 736 * 4) * seconds_per_coalition,
        "Eine Erklärung, Budget 1000 (alle 4 Werte, alle Methoden)": 1000 * seconds_per_coalition,
        "Kohorte: 37 Bilder x Budget 1000": 37 * 1000 * seconds_per_coalition,
    }
    print("\nGeschätzte Laufzeiten:")
    for label, seconds in estimates.items():
        print(f"  {label:60s} {seconds / 60:7.1f} min")


if __name__ == "__main__":
    main()
