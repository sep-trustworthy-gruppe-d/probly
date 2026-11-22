"""Shared fixtures for the ``vision_uncertainty`` tests.

The synthetic "conflict" model has ``M=2`` ensemble members over ``K=2`` classes. The members
only disagree when *both* marker patches (top-left and bottom-right of a 2x2 grid) are visible,
so epistemic uncertainty is a pure pairwise interaction between those two patches.
"""

from __future__ import annotations

# TODO(upstream-pr): remove Windows cext workaround
import sys
import types
from pathlib import Path

# Make sibling modules importable when pytest is launched from the project root.
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

try:
    import shapiq.tree.conversion.cext  # noqa: F401
except ImportError:
    _stub = types.ModuleType("shapiq.tree.conversion.cext")
    _stub.create_edge_tree_arrays = None  # type: ignore[attr-defined]
    sys.modules["shapiq.tree.conversion.cext"] = _stub

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vision_uncertainty import PatchGrid  # noqa: E402

SIZE = 16
CONFLICT_STRENGTH = 0.4  # members predict 0.5 +/- 0.4 when in conflict


def _patch_present(images: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    return images[:, rows, cols, :].mean(axis=(1, 2, 3)) > 0.5


def make_conflict_predict_fn(size: int = SIZE):
    """Ensemble that disagrees only if patch 0 (top-left) AND patch 3 (bottom-right) are visible."""
    half = size // 2

    def predict_fn(images: np.ndarray) -> np.ndarray:
        a = _patch_present(images, slice(0, half), slice(0, half))
        b = _patch_present(images, slice(half, size), slice(half, size))
        conflict = (a & b).astype(float)
        p_pos = 0.5 + CONFLICT_STRENGTH * conflict[:, None] * np.array([[1.0, -1.0]])  # (B, 2)
        return np.stack([p_pos, 1.0 - p_pos], axis=-1)  # (B, M=2, K=2)

    return predict_fn


@pytest.fixture
def conflict_image() -> np.ndarray:
    """Black image whose top-left and bottom-right quadrants are bright ("dog" and "cat")."""
    img = np.zeros((SIZE, SIZE, 3))
    img[: SIZE // 2, : SIZE // 2] = 1.0
    img[SIZE // 2 :, SIZE // 2 :] = 1.0
    return img


@pytest.fixture
def conflict_predict_fn():
    return make_conflict_predict_fn()


@pytest.fixture
def grid_2x2() -> PatchGrid:
    return PatchGrid(SIZE, SIZE, 2, 2)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
