"""Shared test fixtures: tiny random transformers, so the tests run in seconds without downloads."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import timm
import torch

# Make the package importable when pytest is started from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vit_uncertainty.backbones import FrozenViT
from vit_uncertainty.probly_bridge import FeatureLayout, build_methods, make_head

N_CLASSES = 5


def tiny_backbone(name: str, patch_size: int, seed: int) -> FrozenViT:
    """A random 2-layer ViT with the real interface (224 px input, CLS token)."""
    torch.manual_seed(seed)
    model = timm.models.vision_transformer.VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=32, depth=2, num_heads=2, num_classes=0
    )
    return FrozenViT(name, model=model, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


@pytest.fixture(scope="session")
def backbones() -> dict:
    """Two tiny backbones with the patch sizes of ViT-S/16 and DINOv2-S/14."""
    return {"vit": tiny_backbone("vit", 16, 0), "dinov2": tiny_backbone("dinov2", 14, 1)}


@pytest.fixture(scope="session")
def methods(backbones) -> dict:
    """The three probly methods with random (untrained) heads."""
    layout = FeatureLayout.from_backbones(backbones)
    states = {name: [make_head(32, N_CLASSES, seed=s).state_dict() for s in range(6)] for name in backbones}
    return build_methods(layout, states)


@pytest.fixture(scope="session")
def image() -> np.ndarray:
    """A smooth random image (SLIC finds proper segments in it, unlike in pure noise)."""
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, size=(8, 8, 3)).astype(np.uint8)
    return np.kron(small, np.ones((28, 28, 1), dtype=np.uint8))
