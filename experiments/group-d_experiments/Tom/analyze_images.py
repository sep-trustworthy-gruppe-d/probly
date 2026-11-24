"""Compare grid and SLIC explanations for the Tom image examples."""

from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import VGG16_Weights

from vgg16_probly_bridge import build_probly_vgg16, load_image, make_predict_fn
from vision_uncertainty import explain_uncertainty, uncertainty_decomposition

T, P_DROP, BUDGET, BATCH, SEED = 15, 0.3, 1000, 16, 42
N_SEGMENTS, COMPACTNESS, GRID_SHAPE = 16, 10.0, (4, 4)
IMAGES = {
    "Golden Retriever": "GoldenRetriever.jpeg",
    "Hund + Katze": "DogCat.jpeg",
    "Verschwommener Frosch": "blurryFrog.jpeg",
}
METHODS = ("grid", "slic")
MEASURES = ("target_probability", "total")


def main() -> None:
    """Run both segmentations, print a comparison, and save the heat-map figure."""
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_probly_vgg16(kind="dropout", p=P_DROP)
    predict_fn = make_predict_fn(model, n_samples=T, device=device)
    categories = VGG16_Weights.DEFAULT.meta["categories"]
    picture_dir = Path(__file__).resolve().parent / "Pictures"
    data = {}

    for name, filename in IMAGES.items():
        image = load_image(str(picture_dir / filename))
        probabilities = predict_fn(image[None])[0]
        target = int(probabilities.mean(axis=0).argmax())
        total, aleatoric, epistemic = uncertainty_decomposition(probabilities[None])
        target_probs = probabilities[:, target]
        data[name] = {
            "image": image,
            "target": target,
            "confidence": float(target_probs.mean()),
            "entropy": (float(total[0]), float(aleatoric[0]), float(epistemic[0])),
            "cache": {method: {} for method in METHODS},
        }

    for name, result in data.items():
        result["explanations"] = {}
        for method in METHODS:
            started = time.perf_counter()
            explanations = {}
            for measure in MEASURES:
                explanations[measure] = explain_uncertainty(
                    result["image"],
                    predict_fn,
                    grid_shape=GRID_SHAPE,
                    measure=measure,
                    target_class=result["target"] if measure == "target_probability" else None,
                    baseline=0.5,
                    budget=BUDGET,
                    batch_size=BATCH,
                    random_state=SEED,
                    cache=result["cache"][method],
                    segmentation=method,
                    n_segments=N_SEGMENTS,
                    compactness=COMPACTNESS,
                )
            result["explanations"][method] = explanations
            result.setdefault("players", {})[method] = explanations["total"].grid.n_players
            result.setdefault("runtime", {})[method] = time.perf_counter() - started
            print(f"{name}: {method} abgeschlossen ({result['runtime'][method]:.1f}s)", flush=True)

    print(f"Device={device} | T={T} | p={P_DROP} | budget={BUDGET} | seed={SEED}")
    print("Bild | Methode | N | Top-1 / p | H total / alea / epi | φP min..max | φH min..max | s")
    for name, result in data.items():
        total, aleatoric, epistemic = result["entropy"]
        for method in METHODS:
            prob = result["explanations"][method]["target_probability"].first_order
            entropy = result["explanations"][method]["total"].first_order
            print(
                f"{name} | {method} | {result['players'][method]} | "
                f"{categories[result['target']]} {result['confidence']:.3f} | "
                f"{total:.3f}/{aleatoric:.3f}/{epistemic:.3f} | "
                f"{prob.min():+.3f}..{prob.max():+.3f} | "
                f"{entropy.min():+.3f}..{entropy.max():+.3f} | "
                f"{result['runtime'][method]:.1f}"
            )

    fig, axes = plt.subplots(len(data), 4, figsize=(16, 3.8 * len(data)), squeeze=False)
    columns = (("grid", "target_probability"), ("grid", "total"), ("slic", "target_probability"), ("slic", "total"))
    titles = ("Grid: probability", "Grid: entropy", "SLIC: probability", "SLIC: entropy")
    for row, (name, result) in enumerate(data.items()):
        limits = {
            measure: max(
                max(float(np.abs(result["explanations"][method][measure].first_order).max()) for method in METHODS),
                1e-12,
            )
            for measure in MEASURES
        }
        for axis, (method, measure), title in zip(axes[row], columns, titles, strict=True):
            explanation = result["explanations"][method][measure]
            labels = explanation.grid.label_map()
            axis.imshow(result["image"])
            axis.imshow(
                explanation.first_order_image(),
                cmap="bwr",
                vmin=-limits[measure],
                vmax=limits[measure],
                alpha=0.55,
            )
            if explanation.grid.n_players > 1:
                axis.contour(
                    labels, levels=np.arange(explanation.grid.n_players - 1) + 0.5, colors="white", linewidths=0.45
                )
            axis.set_title(f"{name}\n{title} (N={explanation.grid.n_players})", fontsize=9)
            axis.axis("off")

    fig.tight_layout()
    output = Path(__file__).resolve().parent / "grid_vs_slic_comparison.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVergleichsgrafik gespeichert: {output}")


if __name__ == "__main__":
    main()
