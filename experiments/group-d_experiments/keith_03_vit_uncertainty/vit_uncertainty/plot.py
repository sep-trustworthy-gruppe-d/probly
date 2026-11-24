"""Plots that show prediction and uncertainty side by side.

The main figure, :func:`plot_prediction_and_uncertainty`, answers the project question for one image
in one row: what does the model predict, which regions support that prediction (P), and which
regions make it uncertain, split into noise (AU) and missing knowledge (EU)?
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np

from .game import QUANTITY_NAMES_DE

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .explainer import UncertaintyExplanation

SHORT_TITLES = {"P": "Vorhersage (P)", "TU": "Gesamt (TU)", "AU": "Rauschen (AU)", "EU": "fehlendes Wissen (EU)"}


def region_boundaries(regions: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels at the border between two players."""
    border = np.zeros(regions.shape, dtype=bool)
    border[:, 1:] |= regions[:, 1:] != regions[:, :-1]
    border[1:, :] |= regions[1:, :] != regions[:-1, :]
    return border


def show_values(
    ax: Axes,
    image: np.ndarray,
    regions: np.ndarray,
    values: np.ndarray,
    *,
    vmax: float | None = None,
    title: str = "",
    alpha: float = 0.6,
) -> None:
    """Overlay one value per region on the image (red = raises the value, blue = lowers it)."""
    pixel_values = values[regions]
    vmax = float(np.abs(values).max()) if vmax is None else vmax
    vmax = vmax if vmax > 0 else 1e-12
    ax.imshow(image)
    ax.imshow(pixel_values, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -vmax, vmax), alpha=alpha)
    border = region_boundaries(regions)
    overlay = np.zeros((*regions.shape, 4))
    overlay[border] = (1, 1, 1, 0.8)
    ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_prediction_and_uncertainty(
    explanation: UncertaintyExplanation,
    *,
    quantities: tuple[str, ...] = ("P", "TU", "AU", "EU"),
    title: str | None = None,
) -> Figure:
    """One row: the image with its prediction, then one heatmap per quantity.

    Every heatmap shows the Shapley value of each region and has its own symmetric colour scale. Below
    the title are ``v(N)`` (full image) and ``v(∅)`` (everything hidden); the values of a map add up to
    ``v(N) - v(∅)``.

    Args:
        explanation: :class:`~vit_uncertainty.explainer.UncertaintyExplanation`.
        quantities: Which maps to show.
        title: Optional figure title.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(quantities) + 1, figsize=(3.1 * (len(quantities) + 1), 3.6), layout="constrained")
    axes[0].imshow(explanation.image)
    border = region_boundaries(explanation.regions)
    overlay = np.zeros((*border.shape, 4))
    overlay[border] = (1, 1, 1, 0.9)
    axes[0].imshow(overlay)
    name = explanation.class_name or f"Klasse {explanation.target_class}"
    axes[0].set_title(f"Vorhersage: {name}\nP = {explanation.full['P']:.0%} · {explanation.method}", fontsize=9)
    axes[0].axis("off")
    for ax, quantity in zip(axes[1:], quantities, strict=True):
        show_values(
            ax,
            explanation.image,
            explanation.regions,
            explanation.first_order(quantity),
            title=(
                f"{SHORT_TITLES[quantity]}\n"
                f"v(N) = {explanation.full[quantity]:.3f}, v(∅) = {explanation.empty[quantity]:.3f}"
            ),
        )
    if title:
        fig.suptitle(title)
    return fig


def plot_pairs(
    explanation: UncertaintyExplanation, quantity: str = "EU", *, top: int = 3, ax: Axes | None = None
) -> Axes:
    """Show the strongest pair interactions of one quantity as lines between region centres.

    A red line: the two regions together *raise* the value more than each alone; blue: less.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.6), layout="constrained")
    pairs = explanation.pairs(quantity)
    ax.imshow(explanation.image)
    border = region_boundaries(explanation.regions)
    overlay = np.zeros((*border.shape, 4))
    overlay[border] = (1, 1, 1, 0.9)
    ax.imshow(overlay)
    centres = [np.argwhere(explanation.regions == player).mean(axis=0) for player in range(explanation.n_players)]
    order = sorted(
        ((abs(pairs[i, j]), i, j) for i in range(len(pairs)) for j in range(i + 1, len(pairs))), reverse=True
    )[:top]
    strongest = order[0][0] if order else 1.0
    for value, i, j in order:
        colour = "tab:red" if pairs[i, j] > 0 else "tab:blue"
        (y1, x1), (y2, x2) = centres[i], centres[j]
        ax.plot([x1, x2], [y1, y2], color=colour, linewidth=1 + 5 * value / (strongest + 1e-12))
    ax.set_title(f"Stärkste Paare · {QUANTITY_NAMES_DE[quantity]}", fontsize=9)
    ax.axis("off")
    return ax


def plot_comparison(
    explanations: dict[str, UncertaintyExplanation], quantity: str = "EU", *, shared_scale: bool = False
) -> Figure:
    """Compare one quantity across several explanations of the same image (methods, players, removal)."""
    fig, axes = plt.subplots(1, len(explanations), figsize=(3.1 * len(explanations), 3.4), layout="constrained")
    axes = np.atleast_1d(axes)
    vmax = max(np.abs(e.first_order(quantity)).max() for e in explanations.values()) if shared_scale else None
    for ax, (label, explanation) in zip(axes, explanations.items(), strict=True):
        show_values(
            ax,
            explanation.image,
            explanation.regions,
            explanation.first_order(quantity),
            vmax=vmax,
            title=f"{label}\n{SHORT_TITLES[quantity]}, v(N) = {explanation.full[quantity]:.3f}",
        )
    return fig
