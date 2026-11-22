"""Tests for :mod:`vision_uncertainty` (pure numpy; no torch / probly needed)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import vision_uncertainty as vu
from conftest import CONFLICT_STRENGTH, SIZE
from vision_uncertainty import (
    PatchGrid,
    UncertaintyGame,
    credal_hull_volume,
    credal_width,
    explain_uncertainty,
    mask_image,
    predictive_entropy,
    uncertainty_decomposition,
    upsample_to_image,
)

H_CONFLICT = float(
    predictive_entropy(np.array([0.5, 0.5]))
    - predictive_entropy(np.array([[0.9, 0.1], [0.1, 0.9]])).mean()
)


# ----------------------------- hygiene ----------------------------------- #
def test_module_has_no_heavy_imports():
    tree = ast.parse(Path(vu.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    forbidden = {"torch", "probly", "jax", "sklearn", "PIL", "cv2", "torchvision"}
    assert not roots & forbidden
    allowed = {"__future__", "sys", "types", "dataclasses", "typing", "numpy", "scipy", "shapiq"}
    assert roots <= allowed


# ----------------------------- measures ---------------------------------- #
def test_entropy_uniform_and_deterministic():
    assert predictive_entropy(np.array([0.5, 0.5])) == pytest.approx(np.log(2))
    assert predictive_entropy(np.array([1.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_decomposition_identity_and_identical_members(rng):
    probs = rng.dirichlet(np.ones(4), size=(5, 6))  # (5 inputs, 6 members, 4 classes)
    total, alea, epi = uncertainty_decomposition(probs)
    np.testing.assert_allclose(total, alea + epi, atol=1e-9)
    same = np.repeat(probs[:, :1], 6, axis=1)
    _, _, epi_same = uncertainty_decomposition(same)
    np.testing.assert_allclose(epi_same, 0.0, atol=1e-9)


def test_decomposition_disagreement_is_epistemic():
    probs = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    total, alea, epi = uncertainty_decomposition(probs)
    assert epi[0] == pytest.approx(H_CONFLICT)
    assert total[0] > alea[0] > 0


def test_decomposition_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        uncertainty_decomposition(np.array([0.5, 0.5]))


def test_credal_width():
    probs = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]])
    np.testing.assert_allclose(credal_width(probs), [1.6, 0.0])


def test_credal_hull_volume(rng):
    assert credal_hull_volume(np.tile([0.3, 0.7], (5, 1))) == 0.0  # degenerate
    assert credal_hull_volume(np.array([[0.5, 0.5]])) == 0.0  # too few points
    assert credal_hull_volume(np.array([[0.2, 0.8], [0.6, 0.4]])) == pytest.approx(0.4)  # K=2 interval
    pts = rng.dirichlet(np.ones(4), size=30)
    assert credal_hull_volume(pts) > 0
    with pytest.raises(ValueError, match="shape"):
        credal_hull_volume(np.ones(3))


# ----------------------------- grid & masking ---------------------------- #
@pytest.mark.parametrize(("h", "w", "r", "c"), [(16, 16, 4, 4), (30, 25, 4, 3), (7, 7, 7, 7)])
def test_grid_partitions_image_exactly(h, w, r, c):
    grid = PatchGrid(h, w, r, c)
    assert grid.n_players == r * c
    cover = np.zeros((h, w), dtype=int)
    for rs, cs in grid.slices:
        cover[rs, cs] += 1
    assert (cover == 1).all()


def test_grid_invalid():
    with pytest.raises(ValueError, match="Grid"):
        PatchGrid(4, 4, 5, 1)
    with pytest.raises(ValueError, match="Grid"):
        PatchGrid(4, 4, 0, 1)


def test_mask_image_keeps_only_coalition(conflict_image, grid_2x2):
    baseline = np.full_like(conflict_image, 0.25)
    out = mask_image(conflict_image, grid_2x2, np.array([True, False, False, False]), baseline)
    half = SIZE // 2
    assert (out[:half, :half] == 1.0).all()
    assert (out[half:, half:] == 0.25).all()
    assert (conflict_image[half:, half:] == 1.0).all()  # input untouched


def test_upsample_to_image(grid_2x2):
    canvas = upsample_to_image(np.array([[1, 2], [3, 4]]), grid_2x2)
    assert canvas.shape == (SIZE, SIZE)
    assert canvas[0, 0] == 1 and canvas[0, -1] == 2 and canvas[-1, 0] == 3 and canvas[-1, -1] == 4
    with pytest.raises(ValueError, match="Expected shape"):
        upsample_to_image(np.zeros((3, 3)), grid_2x2)


# ----------------------------- baselines --------------------------------- #
@pytest.mark.parametrize("baseline", ["zeros", "mean", "blur", 0.3, "array"])
def test_baselines_run(conflict_image, conflict_predict_fn, baseline):
    if baseline == "array":
        baseline = np.zeros_like(conflict_image)
    exp = explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline=baseline)
    assert exp.first_order.shape == (2, 2)


def test_invalid_baseline(conflict_image, conflict_predict_fn):
    with pytest.raises(ValueError, match="Unknown baseline"):
        explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline="nope")
    with pytest.raises(ValueError, match="shape"):
        explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline=np.zeros((2, 2)))


# ----------------------------- game -------------------------------------- #
def test_game_values_and_normalisation(conflict_image, conflict_predict_fn, grid_2x2):
    game = UncertaintyGame(conflict_image, conflict_predict_fn, grid_2x2, baseline="zeros")
    coalitions = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)
    np.testing.assert_allclose(game(coalitions), [0, 0, H_CONFLICT, H_CONFLICT], atol=1e-9)


def test_game_batching_is_consistent(conflict_image, conflict_predict_fn, grid_2x2):
    coalitions = np.array([[(i >> b) & 1 for b in range(4)] for i in range(16)], dtype=bool)
    g1 = UncertaintyGame(conflict_image, conflict_predict_fn, grid_2x2, baseline="zeros", batch_size=1)
    g2 = UncertaintyGame(conflict_image, conflict_predict_fn, grid_2x2, baseline="zeros", batch_size=7)
    np.testing.assert_allclose(g1(coalitions), g2(coalitions))


@pytest.mark.parametrize("measure", ["total", "aleatoric", "epistemic", "credal_width", "credal_volume"])
def test_all_measures_finite(conflict_image, conflict_predict_fn, grid_2x2, measure):
    game = UncertaintyGame(conflict_image, conflict_predict_fn, grid_2x2, measure=measure, baseline="zeros")
    assert np.isfinite(game(np.ones((1, 4), dtype=bool))).all()


def test_game_validation(conflict_image, conflict_predict_fn, grid_2x2):
    with pytest.raises(ValueError, match="Unknown uncertainty"):
        UncertaintyGame(conflict_image, conflict_predict_fn, grid_2x2, measure="bogus")
    with pytest.raises(ValueError, match="batch_size"):
        UncertaintyGame(conflict_image, conflict_predict_fn, grid_2x2, batch_size=0)
    with pytest.raises(ValueError, match="(H, W, C)"):
        UncertaintyGame(conflict_image[..., 0], conflict_predict_fn, grid_2x2)
    with pytest.raises(ValueError, match="grid"):
        UncertaintyGame(conflict_image, conflict_predict_fn, PatchGrid(8, 8, 2, 2))


def test_predict_fn_wrong_shape_raises(conflict_image, grid_2x2):
    def bad(images):
        return np.full((len(images), 2), 0.5)  # missing member axis

    with pytest.raises(ValueError, match=r"\(B, M, K\)"):
        UncertaintyGame(conflict_image, bad, grid_2x2)


# ----------------------------- explanation ------------------------------- #
def test_pure_interaction_is_isolated(conflict_image, conflict_predict_fn):
    """k-SII puts all epistemic uncertainty on the (0, 3) pair and ~0 on single patches."""
    exp = explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline="zeros")
    assert exp.exact
    np.testing.assert_allclose(exp.first_order, 0.0, atol=1e-9)
    assert exp.interactions[0, 3] == pytest.approx(H_CONFLICT, rel=1e-6)
    i, j, val = exp.top_interactions(1)[0]
    assert (i, j) == (0, 3) and val == pytest.approx(H_CONFLICT, rel=1e-6)
    np.testing.assert_allclose(exp.interactions, exp.interactions.T)
    assert np.allclose(np.diag(exp.interactions), 0.0)
    assert exp.interaction_map(0)[1, 1] == pytest.approx(H_CONFLICT, rel=1e-6)


def test_efficiency(conflict_image, conflict_predict_fn):
    exp = explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline="zeros")
    total = exp.first_order.sum() + np.triu(exp.interactions, 1).sum()
    assert total == pytest.approx(exp.full_value, rel=1e-6)
    assert exp.full_value == pytest.approx(H_CONFLICT, rel=1e-6)


def test_no_conflict_means_no_attribution(conflict_predict_fn):
    img = np.zeros((SIZE, SIZE, 3))
    img[: SIZE // 2, : SIZE // 2] = 1.0  # only one marker
    exp = explain_uncertainty(img, conflict_predict_fn, grid_shape=(2, 2), baseline="zeros")
    assert exp.full_value == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(exp.interactions, 0.0, atol=1e-9)


def test_budgeted_approximation_finds_corner_conflict():
    """4x4 grid (2**16 coalitions): a 2000-call budget triggers the approximator."""
    img = np.ones((16, 16, 3))

    def corner_conflict(images: np.ndarray) -> np.ndarray:
        a = images[:, :4, :4].mean(axis=(1, 2, 3)) > 0.5
        b = images[:, 12:, 12:].mean(axis=(1, 2, 3)) > 0.5
        p = 0.5 + 0.4 * (a & b)[:, None] * np.array([[1.0, -1.0]])
        return np.stack([p, 1.0 - p], axis=-1)

    approx = explain_uncertainty(
        img, corner_conflict, grid_shape=(4, 4), baseline="zeros", budget=2000, random_state=1
    )
    assert not approx.exact
    i, j, val = approx.top_interactions(1)[0]
    assert (i, j) == (0, 15)
    assert val == pytest.approx(H_CONFLICT, rel=0.05)
    assert approx.first_order.shape == (4, 4)
    assert abs(approx.first_order[0, 0]) < 0.05 * H_CONFLICT


def test_budget_respects_model_call_limit(conflict_image, conflict_predict_fn):
    calls = {"n": 0}

    def counting(images):
        calls["n"] += len(images)
        return conflict_predict_fn(images)

    explain_uncertainty(conflict_image, counting, grid_shape=(4, 4), baseline="zeros", budget=500, random_state=0)
    assert calls["n"] <= 500 + 2  # budget + empty-coalition normalisation (+ full value)


def test_reproducible_with_seed(conflict_image, conflict_predict_fn):
    kw = {"grid_shape": (4, 4), "baseline": "zeros", "budget": 400, "random_state": 3}
    a = explain_uncertainty(conflict_image, conflict_predict_fn, **kw)
    b = explain_uncertainty(conflict_image, conflict_predict_fn, **kw)
    np.testing.assert_allclose(a.interactions, b.interactions)


def test_top_interactions_ordering(conflict_image, conflict_predict_fn):
    exp = explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline="zeros")
    vals = [abs(v) for *_, v in exp.top_interactions(6)]
    assert vals == sorted(vals, reverse=True)
    assert len(exp.top_interactions(100)) == 6  # C(4, 2)


def test_first_order_image_shape(conflict_image, conflict_predict_fn):
    exp = explain_uncertainty(conflict_image, conflict_predict_fn, grid_shape=(2, 2), baseline="zeros")
    assert exp.first_order_image().shape == (SIZE, SIZE)


def test_conflict_strength_constant_sane():
    assert 0 < CONFLICT_STRENGTH < 0.5
