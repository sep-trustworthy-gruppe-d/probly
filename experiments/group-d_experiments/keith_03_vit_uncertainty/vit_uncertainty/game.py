"""The uncertainty game: what a shapiq game is when the model is a probly predictor.

For an image with players ``N`` (superpixels or grid cells) and a coalition ``S`` of visible players:

    v_P(S)  = mean probability of the target class,        when only S is visible
    v_TU(S) = total uncertainty of the probly predictor,   when only S is visible
    v_AU(S) = aleatoric part (expected entropy)
    v_EU(S) = epistemic part (mutual information)

``TU = AU + EU`` holds for every coalition (probly's entropy decomposition). Shapley values and
interaction indices are linear in the game, so the explanations inherit it:
``phi(TU) = phi(AU) + phi(EU)``. ``tests/test_game.py`` checks this.

Reading the values: a positive Shapley value for EU means "this region *adds* missing knowledge
compared with the image where it is hidden"; a negative value means "seeing this region reduces
the model's missing knowledge".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapiq

from .probly_bridge import uncertainty
from .removal import coalition_codes

if TYPE_CHECKING:
    from torch import nn

    from .removal import CoalitionFeatures

QUANTITIES = ("P", "TU", "AU", "EU")
QUANTITY_NAMES_DE = {
    "P": "Wahrscheinlichkeit der Zielklasse",
    "TU": "Gesamtunsicherheit (TU)",
    "AU": "aleatorisch, Rauschen (AU)",
    "EU": "epistemisch, fehlendes Wissen (EU)",
}


class UncertaintyGame(shapiq.Game):
    """A shapiq game whose value is P, TU, AU, or EU of a probly predictor.

    Args:
        coalition_features: Evaluates coalitions of the explained image (shared, cached).
        predictor: Any probly predictor built by :mod:`vit_uncertainty.probly_bridge`.
        quantity: ``"P"``, ``"TU"``, ``"AU"``, or ``"EU"``.
        target_class: Class whose probability ``"P"`` explains (ignored for the uncertainties).
        num_samples: Forward samples for MC Dropout.
        seed: Seed of the MC Dropout masks (the same masks for every coalition).
        score_cache: Optional dictionary shared by the four games of one image and one predictor.
            It stores probabilities and TU/AU/EU per coalition, so probly runs once per coalition
            instead of once per coalition and game.
    """

    def __init__(
        self,
        coalition_features: CoalitionFeatures,
        predictor: nn.Module,
        quantity: str,
        *,
        target_class: int = 0,
        num_samples: int = 50,
        seed: int = 0,
        score_cache: dict | None = None,
    ) -> None:
        """Remember the coalition evaluator, the predictor, and which quantity to return."""
        if quantity not in QUANTITIES:
            msg = f"Unknown quantity {quantity!r}; use one of {QUANTITIES}."
            raise ValueError(msg)
        self.coalition_features = coalition_features
        self.predictor = predictor
        self.quantity = quantity
        self.target_class = target_class
        self.num_samples, self.seed = num_samples, seed
        self.score_cache = {} if score_cache is None else score_cache
        # normalize=False keeps v(empty) visible as `baseline_value`; it is part of the story
        # (how uncertain is the model when every player is hidden?).
        super().__init__(n_players=coalition_features.n_players, normalize=False)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """One value per coalition (row of the boolean matrix ``coalitions``)."""
        codes = [int(code) for code in coalition_codes(coalitions)]
        missing = list(dict.fromkeys(code for code in codes if code not in self.score_cache))
        if missing:
            rows = {code: index for index, code in enumerate(codes)}
            features = self.coalition_features(np.stack([coalitions[rows[code]] for code in missing]))
            scores = uncertainty(self.predictor, features, num_samples=self.num_samples, seed=self.seed)
            for i, code in enumerate(missing):
                self.score_cache[code] = {
                    "P": scores["probabilities"][i],
                    "TU": scores["TU"][i],
                    "AU": scores["AU"][i],
                    "EU": scores["EU"][i],
                }
        if self.quantity == "P":
            return np.array([self.score_cache[code]["P"][self.target_class] for code in codes])
        return np.array([self.score_cache[code][self.quantity] for code in codes])
