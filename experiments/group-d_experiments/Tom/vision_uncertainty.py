"""Explain 1st- and 2nd-order predictive uncertainty on images with ``shapiq``.

Idea
----
On ambiguous or out-of-distribution images (e.g. one containing both a dog and
a cat) single patches may produce confident predictions, while the
*combination* of two conflicting patches makes an ensemble / credal set
disagree. Standard Shapley values spread such a pure interaction over the
participating players; ``k-SII`` with ``max_order=2`` separates the individual
effect ``phi_i`` from the pairwise conflict ``phi_{i,j}``.

Design
------
* Players are the cells of a regular ``rows x cols`` patch grid.
* A coalition keeps its patches and replaces all other patches by a baseline.
* The model is any callable ``predict_fn(images) -> probs`` with
  ``images`` of shape ``(B, H, W, C)`` and ``probs`` of shape
  ``(B, M, K)``: ``M`` probability vectors (ensemble members, MC-dropout
  samples, credal vertices, ...) over ``K`` classes per image.
* The game value is a scalar uncertainty measure of those ``M`` vectors.

This module is model-agnostic: it only depends on the standard library,
``numpy``, ``scipy`` and ``shapiq`` (no torch / probly / PIL / sklearn).
"""

from __future__ import annotations

# TODO(upstream-pr): remove Windows cext workaround
import sys
import types

try:
    import shapiq.tree.conversion.cext  # noqa: F401
except ImportError:
    _stub = types.ModuleType("shapiq.tree.conversion.cext")
    _stub.create_edge_tree_arrays = None  # type: ignore[attr-defined]
    sys.modules["shapiq.tree.conversion.cext"] = _stub

from dataclasses import dataclass, field  # noqa: E402
from typing import Callable, Literal  # noqa: E402

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial import ConvexHull, QhullError  # noqa: E402

import shapiq  # noqa: E402

__all__ = [
    "PatchGrid",
    "UncertaintyExplanation",
    "UncertaintyGame",
    "UncertaintyMeasure",
    "credal_hull_volume",
    "credal_width",
    "explain_uncertainty",
    "predictive_entropy",
    "uncertainty_decomposition",
    "upsample_to_image",
]

UncertaintyMeasure = Literal[
    "total", "aleatoric", "epistemic", "credal_width", "credal_volume", "target_probability"
]
PredictFn = Callable[[np.ndarray], np.ndarray]
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Uncertainty measures                                                        #
# --------------------------------------------------------------------------- #
def predictive_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy (nats) of probability vectors along ``axis``."""
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def uncertainty_decomposition(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entropy-based decomposition ``total = aleatoric + epistemic``.

    Args:
        probs: Array ``(..., M, K)`` of ``M`` probability vectors over ``K`` classes.

    Returns:
        ``(total, aleatoric, epistemic)`` each of shape ``(...)``. ``total`` is the entropy
        of the mean prediction, ``aleatoric`` the mean entropy of the members and
        ``epistemic`` their difference (mutual information, clipped at 0).
    """
    probs = _check_probs(probs)
    total = predictive_entropy(probs.mean(axis=-2))
    aleatoric = predictive_entropy(probs).mean(axis=-1)
    epistemic = np.maximum(total - aleatoric, 0.0)
    return total, aleatoric, epistemic


def credal_width(probs: np.ndarray) -> np.ndarray:
    """Summed class-wise width ``sum_k (max_m p_mk - min_m p_mk)`` of the credal bounds."""
    probs = _check_probs(probs)
    return (probs.max(axis=-2) - probs.min(axis=-2)).sum(axis=-1)


def credal_hull_volume(probs: np.ndarray, max_dims: int = 3) -> float:
    """Volume of the convex hull spanned by ``M`` probability vectors of one input.

    The hull lives in a ``(K-1)``-dimensional simplex. To keep Qhull stable and cheap, the
    ``min(K-1, max_dims)`` most probable classes (by mean probability) are used as coordinates.
    For two classes the hull is an interval and its length is returned. Degenerate
    configurations (too few or collinear points) return ``0.0``.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        msg = f"Expected shape (M, K), got {probs.shape}."
        raise ValueError(msg)
    n_members, n_classes = probs.shape
    dims = min(n_classes - 1, max_dims)
    if dims < 1 or n_members < 2 or (dims > 1 and n_members <= dims):
        return 0.0
    top = np.argsort(-probs.mean(axis=0))[:dims]
    if dims == 1:  # Qhull needs >= 2-D; the "hull" is an interval
        return float(np.ptp(probs[:, top[0]]))
    try:
        return float(ConvexHull(probs[:, top]).volume)
    except QhullError:
        return 0.0


def _check_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim < 2:
        msg = f"Expected probabilities of shape (..., M, K), got {probs.shape}."
        raise ValueError(msg)
    return probs


# --------------------------------------------------------------------------- #
# Patch grid & masking                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatchGrid:
    """Regular ``n_rows x n_cols`` partition of an ``(H, W)`` image; players are row-major."""

    height: int
    width: int
    n_rows: int
    n_cols: int
    slices: tuple[tuple[slice, slice], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (1 <= self.n_rows <= self.height and 1 <= self.n_cols <= self.width):
            msg = "Grid must satisfy 1 <= n_rows <= height and 1 <= n_cols <= width."
            raise ValueError(msg)
        r_edges = np.linspace(0, self.height, self.n_rows + 1).round().astype(int)
        c_edges = np.linspace(0, self.width, self.n_cols + 1).round().astype(int)
        slices = tuple(
            (slice(int(r_edges[r]), int(r_edges[r + 1])), slice(int(c_edges[c]), int(c_edges[c + 1])))
            for r in range(self.n_rows)
            for c in range(self.n_cols)
        )
        object.__setattr__(self, "slices", slices)

    @property
    def n_players(self) -> int:
        return self.n_rows * self.n_cols

    def player_names(self) -> list[str]:
        return [f"patch_{r}_{c}" for r in range(self.n_rows) for c in range(self.n_cols)]

    def to_rc(self, player: int) -> tuple[int, int]:
        """Row / column of a player index."""
        return divmod(player, self.n_cols)


def _make_baseline(image: np.ndarray, baseline: str | float | np.ndarray) -> np.ndarray:
    if isinstance(baseline, str):
        if baseline == "zeros":
            return np.zeros_like(image)
        if baseline == "mean":
            return np.broadcast_to(image.mean(axis=(0, 1), keepdims=True), image.shape).copy()
        if baseline == "blur":
            sigma = max(image.shape[0], image.shape[1]) / 8.0
            return gaussian_filter(image, sigma=(sigma, sigma, 0))
        msg = f"Unknown baseline '{baseline}'. Use 'zeros', 'mean', 'blur', a float or an array."
        raise ValueError(msg)
    arr = np.asarray(baseline, dtype=image.dtype)
    if arr.ndim == 0:
        return np.full_like(image, float(arr))
    if arr.shape != image.shape:
        msg = f"Baseline array shape {arr.shape} does not match image shape {image.shape}."
        raise ValueError(msg)
    return arr


def mask_image(
    image: np.ndarray,
    grid: PatchGrid,
    coalition: np.ndarray,
    baseline_image: np.ndarray,
) -> np.ndarray:
    """Keep patches where ``coalition`` is True, fill the rest from ``baseline_image``."""
    out = baseline_image.copy()
    for keep, (rs, cs) in zip(coalition, grid.slices, strict=True):
        if keep:
            out[rs, cs] = image[rs, cs]
    return out


def upsample_to_image(values: np.ndarray, grid: PatchGrid) -> np.ndarray:
    """Paint a ``(n_rows, n_cols)`` array onto an ``(H, W)`` canvas (for heat-map overlays)."""
    values = np.asarray(values, dtype=float)
    if values.shape != (grid.n_rows, grid.n_cols):
        msg = f"Expected shape {(grid.n_rows, grid.n_cols)}, got {values.shape}."
        raise ValueError(msg)
    canvas = np.zeros((grid.height, grid.width), dtype=float)
    for idx, (rs, cs) in enumerate(grid.slices):
        canvas[rs, cs] = values.flat[idx]
    return canvas


# --------------------------------------------------------------------------- #
# Game                                                                        #
# --------------------------------------------------------------------------- #
class UncertaintyGame(shapiq.Game):
    """Cooperative game ``v(S) = uncertainty(model(image with only patches S visible))``.

    The game is normalised so that ``v(empty) = 0``; consequently attributions sum to
    ``v(N) - v(empty)`` (efficiency).
    """

    def __init__(
        self,
        image: np.ndarray,
        predict_fn: PredictFn,
        grid: PatchGrid,
        measure: UncertaintyMeasure = "epistemic",
        baseline: str | float | np.ndarray = "mean",
        batch_size: int = 16,
        target_class: int | None = None,
        cache: dict[bytes, np.ndarray] | None = None,
    ) -> None:
        image = np.asarray(image, dtype=float)
        if image.ndim != 3:
            msg = f"Expected image of shape (H, W, C), got {image.shape}."
            raise ValueError(msg)
        if image.shape[:2] != (grid.height, grid.width):
            msg = "Image size does not match the patch grid."
            raise ValueError(msg)
        if batch_size < 1:
            msg = "batch_size must be >= 1."
            raise ValueError(msg)
        if measure not in (
            "total",
            "aleatoric",
            "epistemic",
            "credal_width",
            "credal_volume",
            "target_probability",
        ):
            msg = f"Unknown uncertainty measure '{measure}'."
            raise ValueError(msg)
        if measure == "target_probability" and target_class is None:
            msg = "target_class is required when measure='target_probability'."
            raise ValueError(msg)
        self.image = image
        self.predict_fn = predict_fn
        self.grid = grid
        self.measure = measure
        self.batch_size = batch_size
        self.target_class = target_class
        self.cache = {} if cache is None else cache
        self.baseline_image = _make_baseline(image, baseline)
        self.n_model_calls = 0  # number of images passed to predict_fn

        empty = self._evaluate(np.zeros((1, grid.n_players), dtype=bool))[0]
        super().__init__(
            grid.n_players,
            normalize=True,
            normalization_value=float(empty),
            player_names=grid.player_names(),
        )

    def _score(self, probs: np.ndarray) -> np.ndarray:
        if self.measure == "target_probability":
            assert self.target_class is not None
            if not 0 <= self.target_class < probs.shape[-1]:
                msg = f"target_class must be in [0, {probs.shape[-1]}), got {self.target_class}."
                raise ValueError(msg)
            return probs.mean(axis=-2)[:, self.target_class]
        if self.measure == "total":
            return uncertainty_decomposition(probs)[0]
        if self.measure == "aleatoric":
            return uncertainty_decomposition(probs)[1]
        if self.measure == "epistemic":
            return uncertainty_decomposition(probs)[2]
        if self.measure == "credal_width":
            return credal_width(probs)
        return np.array([credal_hull_volume(p) for p in probs])

    def _evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.asarray(coalitions, dtype=bool)
        keys = [coalition.tobytes() for coalition in coalitions]
        rows = {key: coalition for key, coalition in zip(keys, coalitions, strict=True)}
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            batch = np.stack(
                [mask_image(self.image, self.grid, rows[key], self.baseline_image) for key in chunk]
            )
            probs = np.asarray(self.predict_fn(batch), dtype=float)
            if probs.ndim != 3 or probs.shape[0] != len(chunk):
                msg = (
                    "predict_fn must return probabilities of shape (B, M, K); "
                    f"got {probs.shape} for a batch of {len(chunk)}."
                )
                raise ValueError(msg)
            self.n_model_calls += len(chunk)
            for key, prediction in zip(chunk, probs, strict=True):
                self.cache[key] = prediction
        return self._score(np.stack([self.cache[key] for key in keys]))

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return self._evaluate(coalitions)


# --------------------------------------------------------------------------- #
# Explanation                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class UncertaintyExplanation:
    """Result of :func:`explain_uncertainty`.

    Attributes:
        grid: The patch grid (players are row-major).
        measure: Explained uncertainty measure.
        first_order: ``(n_rows, n_cols)`` individual attributions ``phi_i``.
        interactions: Symmetric ``(n, n)`` matrix of pairwise ``phi_{i,j}`` (zero diagonal).
        interaction_values: The raw :class:`shapiq.InteractionValues`.
        full_value: ``v(N) - v(empty)``; equals the sum of all attributions (efficiency).
        exact: Whether values were computed exactly (vs. approximated with a budget).
    """

    grid: PatchGrid
    measure: str
    first_order: np.ndarray
    interactions: np.ndarray
    interaction_values: shapiq.InteractionValues
    full_value: float
    exact: bool

    def top_interactions(self, k: int = 5, *, by: Literal["abs", "positive"] = "abs") -> list[tuple[int, int, float]]:
        """Return the ``k`` strongest pairs ``(i, j, phi_ij)`` with ``i < j``."""
        n = self.grid.n_players
        pairs = [(i, j, float(self.interactions[i, j])) for i in range(n) for j in range(i + 1, n)]
        key = (lambda t: abs(t[2])) if by == "abs" else (lambda t: t[2])
        return sorted(pairs, key=key, reverse=True)[:k]

    def interaction_map(self, player: int) -> np.ndarray:
        """``(n_rows, n_cols)`` map of interactions of one patch with all others."""
        return self.interactions[player].reshape(self.grid.n_rows, self.grid.n_cols)

    def first_order_image(self) -> np.ndarray:
        """First-order attributions painted on an ``(H, W)`` canvas."""
        return upsample_to_image(self.first_order, self.grid)


def explain_uncertainty(
    image: np.ndarray,
    predict_fn: PredictFn,
    *,
    grid_shape: tuple[int, int] = (4, 4),
    measure: UncertaintyMeasure = "epistemic",
    baseline: str | float | np.ndarray = "mean",
    budget: int | None = None,
    batch_size: int = 16,
    random_state: int | None = 0,
    target_class: int | None = None,
    cache: dict[bytes, np.ndarray] | None = None,
) -> UncertaintyExplanation:
    """Explain the uncertainty of ``predict_fn`` on ``image`` with ``k-SII`` (``max_order=2``).

    Args:
        image: ``(H, W, C)`` float array.
        predict_fn: Maps ``(B, H, W, C)`` to probabilities ``(B, M, K)``.
        grid_shape: ``(n_rows, n_cols)`` of the patch grid; players = ``n_rows * n_cols``.
        measure: Which uncertainty to explain (see :data:`UncertaintyMeasure`).
        baseline: ``'zeros'``, ``'mean'``, ``'blur'``, a constant or an array of image shape.
        budget: Number of model evaluations. ``None`` (or ``>= 2**n_players``) computes exact values.
        batch_size: Images per ``predict_fn`` call.
        random_state: Seed for the approximator.
        target_class: Class index required for ``measure="target_probability"``.
        cache: Optional per-image prediction cache, shareable across measures.

    Returns:
        An :class:`UncertaintyExplanation`.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 3:
        msg = f"Expected image of shape (H, W, C), got {image.shape}."
        raise ValueError(msg)
    grid = PatchGrid(image.shape[0], image.shape[1], *grid_shape)
    game = UncertaintyGame(
        image,
        predict_fn,
        grid,
        measure,
        baseline,
        batch_size,
        target_class=target_class,
        cache=cache,
    )
    n = grid.n_players

    exact = budget is None or budget >= 2**n
    if exact:
        iv = shapiq.ExactComputer(game)(index="k-SII", order=2)
    else:
        approximator = shapiq.KernelSHAPIQ(n=n, max_order=2, index="k-SII", random_state=random_state)
        iv = approximator.approximate(budget, game)

    first = np.array([iv[(i,)] for i in range(n)], dtype=float)
    pair = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pair[i, j] = pair[j, i] = iv[(i, j)]

    full = float(game(np.ones((1, n), dtype=bool))[0])
    return UncertaintyExplanation(
        grid=grid,
        measure=measure,
        first_order=first.reshape(grid.n_rows, grid.n_cols),
        interactions=pair,
        interaction_values=iv,
        full_value=full,
        exact=exact,
    )
