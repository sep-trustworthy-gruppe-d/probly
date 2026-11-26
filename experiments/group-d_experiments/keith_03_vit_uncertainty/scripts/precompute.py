"""Precompute every transformer pass the notebooks need (optional, saves time in the notebooks).

The notebooks compute missing data themselves. This script only does the slow part in one go, for
example overnight. It never needs trained heads: shapiq's approximators choose their coalitions
independently of the game values, so a placeholder predictor yields exactly the coalitions the
notebooks will ask for later.

    uv run python experiments/group-d_experiments/keith_03_vit_uncertainty/scripts/precompute.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch import nn

from vit_uncertainty import settings
from vit_uncertainty.backbones import FrozenViT, load_backbones
from vit_uncertainty.data import OxfordPets, default_data_root
from vit_uncertainty.explainer import UncertaintyExplainer
from vit_uncertainty.features import (
    coalition_cache,
    full_image_features,
    masked_training_features,
    save_coalition_cache,
)
from vit_uncertainty.probly_bridge import FeatureLayout, build_methods, make_head

if TYPE_CHECKING:
    from shapiq.explainer.custom_types import ExplainerIndices


def placeholder_predictor(backbones: dict[str, FrozenViT]) -> nn.Module:
    """Random heads with the right shapes; only used to drive the approximators."""
    layout = FeatureLayout.from_backbones(backbones)
    states = {
        name: [make_head(b.feature_dim, 37, seed=s).state_dict() for s in range(6)] for name, b in backbones.items()
    }
    return build_methods(layout, states)["Gemischt"]


def explain_and_store(
    pets: OxfordPets,
    backbones: dict[str, FrozenViT],
    predictor: nn.Module,
    image_id: str,
    *,
    index: ExplainerIndices,
    max_order: int,
    budget: int,
    players: str = "slic",
    removal: str = "token",
    fill: str = "blur",
) -> int:
    """Run one explanation so that its coalitions are evaluated, then store them."""
    cache = coalition_cache(
        pets, image_id, backbones, players=players, n_players=settings.N_PLAYERS, removal=removal, fill=fill
    )
    explainer = UncertaintyExplainer(
        backbones,
        predictor,
        players=players,
        n_players=settings.N_PLAYERS,
        index=index,
        max_order=max_order,
        budget=budget,
        random_state=settings.SEED,
        quantities=("TU",),
    )
    explanation = explainer.explain(cache.image, regions=cache.regions, coalition_features=cache)
    save_coalition_cache(cache)
    return explanation.n_forward


def main() -> None:
    """Run all steps; every step skips what is already on disk."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    pets = OxfordPets(default_data_root())
    backbones = load_backbones()
    start = time.perf_counter()

    def log(text: str) -> None:
        print(f"[{(time.perf_counter() - start) / 60:6.1f} min] {text}", flush=True)

    for split in ("test", "trainval"):
        full_image_features(pets, backbones, split)
    log("Merkmale aller Bilder fertig")
    train, val = pets.train_val_indices
    masked_training_features(pets, backbones, val, name="val", per_image=settings.MASKED_PER_IMAGE_VAL)
    masked_training_features(pets, backbones, train, name="train", per_image=settings.MASKED_PER_IMAGE_TRAIN)
    log("Maskierte Trainingsdaten fertig")

    predictor = placeholder_predictor(backbones)
    test_ids = pets.test[0]
    cohort = [test_ids[i] for i in pets.cohorts["explanation"]]
    subset = [cohort[i] for i in settings.SUBSET_POSITIONS]
    # (label, images, cache settings, index, max_order, budgets)
    jobs: list[tuple[str, list[str], dict[str, str], ExplainerIndices, int, list[int]]] = [
        ("Hauptkonfiguration", cohort, {"players": "slic", "removal": "token"}, "k-SII", 2, [settings.BUDGET])
    ]
    jobs.append(("Raster 4x4", subset, {"players": "grid", "removal": "token"}, "k-SII", 2, [settings.BUDGET]))
    jobs.append(
        (
            "Übermalen (blur)",
            subset,
            {"players": "slic", "removal": "fill", "fill": "blur"},
            "k-SII",
            2,
            [settings.BUDGET],
        )
    )
    jobs.append(
        (
            "KernelSHAP-Referenz",
            subset,
            {"players": "slic", "removal": "token"},
            "SV",
            1,
            [*settings.SV_BUDGETS, settings.SV_REFERENCE_BUDGET],
        )
    )
    for label, images, cache_kwargs, index, order, budgets in jobs:
        for number, image_id in enumerate(images, start=1):
            new = sum(
                explain_and_store(
                    pets, backbones, predictor, image_id, index=index, max_order=order, budget=budget, **cache_kwargs
                )
                for budget in budgets
            )
            log(f"{label}: {number}/{len(images)} {image_id} ({new} neue Koalitionen)")
    log("FERTIG")


if __name__ == "__main__":
    main()
