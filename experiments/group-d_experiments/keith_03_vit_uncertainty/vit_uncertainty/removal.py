"""Evaluate coalitions of one image: hide the absent players and run the frozen backbones once.

This is the expensive part of every explanation (one transformer forward pass per coalition and
backbone). Everything after it, i.e. probly heads and ``quantify``, is cheap. We therefore keep the
two apart:

* :class:`CoalitionFeatures` turns coalitions into concatenated backbone features and caches them,
* the games in :mod:`vit_uncertainty.game` turn features into P, TU, AU or EU.

With the cache, the four games of one image (P, TU, AU, EU) and all probly methods share the same
forward passes. This was the key lesson of the DermaMNIST series: separating "evaluate coalitions"
from "measure uncertainty" makes explanations of uncertainty affordable.

Two ways to remove a player are supported:

* ``"token"`` (default): hide its patch tokens inside the transformer (no fill colour at all),
* ``"fill"``: paint its pixels with a fill (``"blur"`` or ``"mean"``, defined as in ``shapiq_games``),
  as a CNN would need it. We use it only to compare with the token removal (design question D2).
"""

from __future__ import annotations

import numpy as np
import torch

from .players import pixel_coalition_mask, regions_to_tokens, token_coalition_mask
from .probly_bridge import FeatureLayout


def fill_image(image: np.ndarray, fill: str) -> np.ndarray:
    """What shows through where a player is removed in pixel space.

    Same definitions as ``shapiq_games.vision`` (shapiq 1.8): ``"mean"`` is the image's mean colour,
    ``"blur"`` a Gaussian blur with radius ``max(height, width) / 32``.
    """
    from PIL import Image, ImageFilter  # noqa: PLC0415

    if fill == "mean":
        colour = image.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
        return np.broadcast_to(colour, image.shape).copy()
    if fill == "blur":
        return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(max(image.shape[:2]) / 32)))
    msg = f"Unknown fill {fill!r}; use 'blur' or 'mean'."
    raise ValueError(msg)


def coalition_codes(coalitions: np.ndarray) -> np.ndarray:
    """One integer per coalition (player ``i`` = bit ``i``); used as cache key."""
    coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
    return coalitions.astype(np.int64) @ (np.int64(1) << np.arange(coalitions.shape[1], dtype=np.int64))


class CoalitionFeatures:
    """Concatenated backbone features of one image for any coalition of its players, cached.

    Args:
        image: ``uint8`` RGB crop ``(224, 224, 3)``.
        regions: Player id per pixel ``(224, 224)``.
        backbones: ``{name: FrozenViT}``; their order defines the feature layout.
        removal: ``"token"`` or ``"fill"``.
        fill: Fill for ``removal="fill"``: ``"blur"`` or ``"mean"``.
        batch_size: Coalitions per forward pass.
    """

    def __init__(
        self,
        image: np.ndarray,
        regions: np.ndarray,
        backbones: dict,
        *,
        removal: str = "token",
        fill: str = "blur",
        batch_size: int = 32,
    ) -> None:
        """Prepare the token maps (or the fill image) of the explained image."""
        if removal not in ("token", "fill"):
            msg = f"Unknown removal {removal!r}; use 'token' or 'fill'."
            raise ValueError(msg)
        self.image, self.regions, self.backbones = image, regions, backbones
        self.removal, self.fill, self.batch_size = removal, fill, batch_size
        self.layout = FeatureLayout.from_backbones(backbones)
        self.n_players = int(regions.max()) + 1
        # For token removal: player id of every patch token, per backbone (patch sizes differ).
        self.token_players = {name: regions_to_tokens(regions, b.patch_size) for name, b in backbones.items()}
        self._filled = fill_image(image, fill) if removal == "fill" else None
        self._cache: dict[int, torch.Tensor] = {}
        self.n_forward = 0  # how many coalitions really went through the backbones

    def _compute(self, coalitions: np.ndarray) -> torch.Tensor:
        blocks = {}
        for name, backbone in self.backbones.items():
            if self.removal == "token":
                visible = token_coalition_mask(coalitions, self.token_players[name])
                images = np.repeat(self.image[None], len(coalitions), axis=0)
                blocks[name] = backbone.features(images, visible, batch_size=self.batch_size)
            else:
                keep = pixel_coalition_mask(coalitions, self.regions)[..., None]
                images = np.where(keep, self.image[None], self._filled[None]).astype(np.uint8)
                blocks[name] = backbone.features(images, batch_size=self.batch_size)
        self.n_forward += len(coalitions)
        return self.layout.concat(blocks)

    def __call__(self, coalitions: np.ndarray) -> torch.Tensor:
        """Features ``(n_coalitions, layout.total)`` in the order of ``coalitions``."""
        coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
        if coalitions.shape[1] != self.n_players:
            msg = f"Expected {self.n_players} players, got {coalitions.shape[1]}."
            raise ValueError(msg)
        codes = coalition_codes(coalitions)
        missing = [i for i, code in enumerate(codes) if int(code) not in self._cache]
        # Evaluate every new coalition only once, even if it appears several times in the request.
        unique_missing = list(dict.fromkeys(int(codes[i]) for i in missing))
        if unique_missing:
            first = {code: i for i, code in reversed(list(enumerate(codes.tolist())))}
            rows = np.stack([coalitions[first[code]] for code in unique_missing])
            for start in range(0, len(rows), 256):
                computed = self._compute(rows[start : start + 256])
                for code, features in zip(unique_missing[start : start + 256], computed, strict=True):
                    self._cache[code] = features
        return torch.stack([self._cache[int(code)] for code in codes])

    def state(self) -> dict:
        """Cached codes and features, e.g. to save them to disk."""
        codes = sorted(self._cache)
        return {"codes": codes, "features": torch.stack([self._cache[c] for c in codes]) if codes else None}

    def load_state(self, state: dict) -> None:
        """Restore a cache saved with :meth:`state`."""
        if state["features"] is not None:
            for code, features in zip(state["codes"], state["features"], strict=True):
                self._cache[int(code)] = features
