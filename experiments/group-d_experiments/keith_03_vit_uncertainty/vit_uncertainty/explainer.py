"""``UncertaintyExplainer``: a shapiq explainer for probly predictors on images.

In shapiq, an explainer is a subclass of :class:`shapiq.Explainer` that implements
``explain_function(x)``: it builds a game for the data point ``x`` and computes interaction values
with shapiq, exactly or with an approximator. Our explainer does this four times per image, for
the prediction (P) and for TU, AU, and EU, and returns all of them together. That is the
"explanation of both" (prediction *and* uncertainty).

Two kinds of values per quantity, from the same coalitions:

* **Shapley values** (``SV``, one value per region). They add up exactly to ``v(N) - v(empty)``, so a
  map of them reads like a budget: "this region lowers the uncertainty by 0.4 nats". The plots use them.
* the requested **interaction index** (default ``k-SII`` up to order 2, as in Tom's comparison): main
  effects plus one value per pair of regions. A pair value says how two regions act *together*.

Cost control:

* The four games share one :class:`~vit_uncertainty.removal.CoalitionFeatures` cache.
* Every game uses the same approximator seed and budget. shapiq's samplers (KernelSHAP for SV and
  KernelSHAPIQ for k-SII) then draw the **same coalitions**, so the transformers run once per coalition,
  not eight times.

Usage::

    explainer = UncertaintyExplainer(backbones, predictor, class_names=pets.classes)
    explanation = explainer.explain(image)          # P, TU, AU, EU as InteractionValues
    plot_prediction_and_uncertainty(explanation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import shapiq
import torch
from torch import nn

from .game import QUANTITIES, UncertaintyGame
from .players import make_regions
from .probly_bridge import uncertainty
from .removal import CoalitionFeatures


@dataclass
class UncertaintyExplanation:
    """Explanations of the prediction and of its uncertainty for one image.

    Attributes:
        image: The explained ``uint8`` crop.
        regions: Player id per pixel.
        values: ``{"P" | "TU" | "AU" | "EU": InteractionValues}`` of the interaction index (e.g. k-SII).
        shapley_values: The same for Shapley values (empty if not requested).
        full: Game value of the full image per quantity, ``v(N)``.
        empty: Game value with every player hidden per quantity, ``v(empty)``.
        target_class: Explained class (prediction on the full image).
        class_name: Its name, if known.
        method: Name of the probly method.
        settings: Players, removal, index, order, budget, seed.
        n_forward: Transformer forward passes (per backbone) used for this explanation.
        seconds: Wall-clock time of the explanation.
    """

    image: np.ndarray
    regions: np.ndarray
    values: dict[str, shapiq.InteractionValues]
    shapley_values: dict[str, shapiq.InteractionValues]
    full: dict[str, float]
    empty: dict[str, float]
    target_class: int
    class_name: str | None = None
    method: str = ""
    settings: dict = field(default_factory=dict)
    n_forward: int = 0
    seconds: float = 0.0

    @property
    def n_players(self) -> int:
        """Number of players (regions)."""
        return int(self.regions.max()) + 1

    def first_order(self, quantity: str) -> np.ndarray:
        """Shapley value of every player (if computed), otherwise the main effects of the interaction index.

        Shapley values add up to ``v(N) - v(empty)``; use them for maps and rankings of regions.
        """
        values = self.shapley_values.get(quantity, self.values[quantity])
        return np.array([values[(player,)] for player in range(self.n_players)])

    def main_effects(self, quantity: str) -> np.ndarray:
        """Order-1 values of the interaction index (they add up to the gap only together with the pairs)."""
        values = self.values[quantity]
        return np.array([values[(player,)] for player in range(self.n_players)])

    def pairs(self, quantity: str) -> np.ndarray:
        """Symmetric matrix of order-2 interactions (zeros if the index has only order 1)."""
        values = self.values[quantity]
        matrix = np.zeros((self.n_players, self.n_players))
        if values.max_order >= 2:
            for i in range(self.n_players):
                for j in range(i + 1, self.n_players):
                    matrix[i, j] = matrix[j, i] = values[(i, j)]
        return matrix

    def pixel_map(self, quantity: str) -> np.ndarray:
        """Spread the main effects over the pixels of their regions (for heatmaps)."""
        return self.first_order(quantity)[self.regions]


class UncertaintyExplainer(shapiq.Explainer):
    """Explain prediction and TU/AU/EU of a probly predictor on images with shapiq.

    Args:
        backbones: ``{name: FrozenViT}``; their order defines the feature layout of ``predictor``.
        predictor: probly predictor from :func:`vit_uncertainty.probly_bridge.build_methods`.
        class_names: Optional class names for the explanation.
        method_name: Optional name of the probly method, stored in the explanation.
        players: ``"slic"`` (default, as in Tom's comparison) or ``"grid"``.
        n_players: Number of players (16 as in Tom's comparison).
        removal: ``"token"`` (hide patch tokens, default) or ``"fill"`` (paint pixels).
        fill: Fill for ``removal="fill"``: ``"blur"`` or ``"mean"``.
        index: shapiq index, e.g. ``"k-SII"`` (default, as Tom) or ``"SV"``.
        max_order: Highest interaction order (2 = pairs of regions).
        budget: Coalitions per game. If ``budget >= 2**n_players`` the values are exact. For k-SII up to
            order 2 it should clearly exceed the number of unknowns ``1 + n + n(n-1)/2`` (137 for 16
            players); Tom used 1000.
        num_samples: MC Dropout samples.
        random_state: Seed of the approximator (the same for all four games).
        quantities: Which of ``("P", "TU", "AU", "EU")`` to explain.
        shapley_values: Also compute Shapley values from the same coalitions (free, default ``True``).
    """

    def __init__(
        self,
        backbones: dict,
        predictor: nn.Module,
        *,
        class_names: list[str] | None = None,
        method_name: str = "",
        players: str = "slic",
        n_players: int = 16,
        removal: str = "token",
        fill: str = "blur",
        index: str = "k-SII",
        max_order: int = 2,
        budget: int = 1000,
        num_samples: int = 50,
        random_state: int = 0,
        quantities: tuple[str, ...] = QUANTITIES,
        shapley_values: bool = True,
    ) -> None:
        """Store the settings; nothing is computed before :meth:`explain`."""
        self.backbones = backbones
        self.predictor = predictor
        self.class_names = class_names
        self.method_name = method_name
        self.players, self.n_players = players, n_players
        self.removal, self.fill = removal, fill
        self.budget, self.num_samples, self.random_state = budget, num_samples, random_state
        self.quantities = tuple(quantities)
        self.with_shapley_values = shapley_values
        # shapiq's base class validates index and max_order and stores the "model": here a function
        # mapping a batch of images to the probly predictor's mean class probabilities.
        super().__init__(model=self.predict_proba, index=index, max_order=max_order)

    # ------------------------------------------------------------------ prediction

    def image_features(self, images: np.ndarray) -> torch.Tensor:
        """Concatenated CLS features of full ``uint8`` images ``(N, 224, 224, 3)``."""
        return torch.cat([b.features(images) for b in self.backbones.values()], dim=1)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Mean class probabilities of the probly predictor for full images."""
        images = np.asarray(images)
        if images.ndim == 3:
            images = images[None]
        return uncertainty(self.predictor, self.image_features(images), num_samples=self.num_samples)["probabilities"]

    # ----------------------------------------------------------------- explanation

    def _interaction_values(
        self, game: UncertaintyGame, budget: int, *, index: str | None = None
    ) -> shapiq.InteractionValues:
        index = self.index if index is None else index
        order = 1 if index == "SV" else self.max_order
        n = game.n_players
        if budget >= 2**n:
            # Few players: evaluate every coalition and compute exact values.
            return shapiq.ExactComputer(game, n_players=n)(index=index, order=order)
        if index == "SV":
            approximator = shapiq.approximator.KernelSHAP(n, random_state=self.random_state)
        else:
            approximator = shapiq.approximator.KernelSHAPIQ(
                n, max_order=order, index=index, random_state=self.random_state
            )
        return approximator.approximate(budget=budget, game=game)

    def explain_function(  # type: ignore[override]
        self,
        x: np.ndarray,
        *,
        regions: np.ndarray | None = None,
        budget: int | None = None,
        coalition_features: CoalitionFeatures | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> UncertaintyExplanation:
        """Explain one image: prediction and TU/AU/EU.

        Args:
            x: ``uint8`` RGB crop ``(224, 224, 3)``.
            regions: Optional player map; by default built with ``players`` and ``n_players``.
            budget: Optional budget overriding the one given at construction.
            coalition_features: Optional existing cache for this image and these regions (lets several
                probly methods share the transformer passes).
            **kwargs: Ignored (compatibility with :meth:`shapiq.Explainer.explain`).

        Returns:
            :class:`UncertaintyExplanation` with one ``InteractionValues`` per quantity.
        """
        start = time.perf_counter()
        image = np.asarray(x)
        regions = make_regions(image, self.players, self.n_players) if regions is None else regions
        cache = coalition_features or CoalitionFeatures(
            image, regions, self.backbones, removal=self.removal, fill=self.fill
        )
        n_before = cache.n_forward
        budget = self.budget if budget is None else budget
        # The explained class is the prediction on the full image (coalition of all players).
        n = int(regions.max()) + 1
        if n < self.max_order:
            msg = f"Only {n} players found; at least max_order={self.max_order} are needed."
            raise ValueError(msg)
        full_scores = uncertainty(
            self.predictor, cache(np.ones((1, n), dtype=bool)), num_samples=self.num_samples, seed=self.random_state
        )
        target = int(full_scores["probabilities"][0].argmax())
        values, shapley, full, empty = {}, {}, {}, {}
        score_cache: dict = {}  # shared by the four games: probly runs once per coalition
        for quantity in self.quantities:
            game = UncertaintyGame(
                cache,
                self.predictor,
                quantity,
                target_class=target,
                num_samples=self.num_samples,
                seed=self.random_state,
                score_cache=score_cache,
            )
            values[quantity] = self._interaction_values(game, budget)
            if self.with_shapley_values:
                # Same budget and seed -> the same coalitions: no extra transformer passes.
                shapley[quantity] = (
                    values[quantity] if self.index == "SV" else self._interaction_values(game, budget, index="SV")
                )
            ends = game(np.stack([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)]))
            empty[quantity], full[quantity] = float(ends[0]), float(ends[1])
        return UncertaintyExplanation(
            image=image,
            regions=regions,
            values=values,
            shapley_values=shapley,
            full=full,
            empty=empty,
            target_class=target,
            class_name=None if self.class_names is None else self.class_names[target],
            method=self.method_name,
            settings={
                "players": self.players,
                "n_players": n,
                "removal": self.removal,
                "fill": self.fill if self.removal == "fill" else None,
                "index": self.index,
                "max_order": self.max_order,
                "budget": budget,
                "random_state": self.random_state,
            },
            n_forward=cache.n_forward - n_before,
            seconds=time.perf_counter() - start,
        )
