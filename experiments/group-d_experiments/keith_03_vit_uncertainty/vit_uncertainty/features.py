"""Compute once, then reuse: backbone features of full and partially hidden images.

All caches live in ``<repo>/artifacts/keith-vit-uncertainty/`` (ignored by git). A cache is only
reused if its manifest (image ids, models, settings) matches exactly; otherwise it is recomputed.

Three kinds of data are cached:

1. **Full-image features** of every image (CLS of both backbones), for training and evaluating the
   probly heads. For the ViT we also keep the final patch tokens (float16); the learned explainer
   (ViT Shapley) reads them.
2. **Masked training features**: for every training and validation image a few random coalitions of
   its 16 SLIC superpixels, hidden in token space. Heads trained on them have seen partial images,
   which is the idea of the "surrogate" model in ViT Shapley (Covert et al., 2023).
3. **Explanation caches** of single images (all coalitions an explanation evaluated).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from .players import make_regions, regions_to_tokens, token_coalition_mask

if TYPE_CHECKING:
    from .data import OxfordPets
    from .removal import CoalitionFeatures

ARTIFACT_DIR = Path(__file__).resolve().parents[4] / "artifacts" / "keith-vit-uncertainty"


def save(path: Path, payload: dict) -> None:
    """Atomically save a payload with ``torch.save``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load(path: Path) -> dict | None:
    """Load a payload saved with :func:`save`, or ``None`` if the file does not exist."""
    return torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else None


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def shapley_kernel_sizes(n_players: int) -> np.ndarray:
    """Probability of each coalition size ``1 .. n-1`` under the Shapley kernel (KernelSHAP)."""
    sizes = np.arange(1, n_players)
    weights = (n_players - 1) / (sizes * (n_players - sizes))
    return weights / weights.sum()


def sample_coalitions(n_players: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw coalitions whose size follows the Shapley kernel, members uniformly at random.

    Small and large coalitions are drawn more often, as in KernelSHAP and ViT Shapley.
    """
    probabilities = shapley_kernel_sizes(n_players)
    sizes = rng.choice(np.arange(1, n_players), size=count, p=probabilities)
    coalitions = np.zeros((count, n_players), dtype=bool)
    for row, size in enumerate(sizes):
        coalitions[row, rng.choice(n_players, size=size, replace=False)] = True
    return coalitions


def full_image_features(
    pets: OxfordPets, backbones: dict, split: str, *, batch_size: int = 32, verbose: bool = True
) -> dict:
    """CLS features of all images of ``"trainval"`` or ``"test"`` (plus ViT patch tokens), cached.

    Returns:
        ``{"ids", "labels", "species", "features": {name: (N, D)}, "vit_tokens": (N, P, D) float16}``.
    """
    ids, labels, species = getattr(pets, split)
    manifest = {
        "schema": 1,
        "split": split,
        "ids": _ids_hash(ids),
        "models": {name: b.model.pretrained_cfg.get("hf_hub_id", name) for name, b in backbones.items()},
    }
    path = ARTIFACT_DIR / f"full_{split}.pt"
    cached = load(path)
    if cached is not None and cached["manifest"] == manifest:
        if verbose:
            print(f"Merkmale '{split}' aus dem Cache geladen ({len(ids)} Bilder).")
        return cached
    start = time.perf_counter()
    features = {name: [] for name in backbones}
    tokens = []
    for first in range(0, len(ids), batch_size):
        crops = np.stack([pets.load_crop(image_id) for image_id in ids[first : first + batch_size]])
        for name, backbone in backbones.items():
            x = backbone.normalize(crops)
            with torch.no_grad():
                all_tokens = backbone.model.forward_features(x)
                features[name].append(backbone.model.forward_head(all_tokens, pre_logits=True).float())
                if name == "vit":
                    tokens.append(all_tokens[:, backbone.n_prefix :].half())
        if verbose and (first // batch_size) % 20 == 0:
            print(f"  {min(first + batch_size, len(ids))}/{len(ids)} Bilder, {time.perf_counter() - start:.0f} s")
    payload = {
        "manifest": manifest,
        "ids": ids,
        "labels": torch.from_numpy(labels),
        "species": torch.from_numpy(species),
        "features": {name: torch.cat(parts) for name, parts in features.items()},
        "vit_tokens": torch.cat(tokens) if tokens else None,
    }
    save(path, payload)
    return payload


def masked_training_features(
    pets: OxfordPets,
    backbones: dict,
    positions: np.ndarray,
    *,
    name: str,
    per_image: int = 8,
    n_players: int = 16,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Features of partially hidden ``trainval`` images (random SLIC coalitions), cached.

    Args:
        pets: :class:`~vit_uncertainty.data.OxfordPets`.
        backbones: ``{name: FrozenViT}``.
        positions: Positions in ``trainval`` (e.g. Dan's training or validation positions).
        name: Cache name, e.g. ``"train"`` or ``"val"``.
        per_image: Random coalitions per image.
        n_players: SLIC superpixels per image.
        seed: Seed of the coalition sampler.
        verbose: Print progress.

    Returns:
        ``{"positions": (M,), "coalitions": (M, n_players), "features": {name: (M, D)},
        "image_positions": (I,), "token_players": {name: (I, n_patches)}}``. Row ``m`` belongs to image
        ``positions[m]``; ``token_players`` says which superpixel each patch token of image ``i`` belongs to
        (needed by the learned explainer).
    """
    ids, _, _ = pets.trainval
    manifest = {
        "schema": 2,
        "positions": _ids_hash([int(p) for p in positions]),
        "per_image": per_image,
        "n_players": n_players,
        "seed": seed,
        "players": "slic",
        "models": {key: b.model.pretrained_cfg.get("hf_hub_id", key) for key, b in backbones.items()},
    }
    path = ARTIFACT_DIR / f"masked_{name}.pt"
    cached = load(path)
    if cached is not None and cached["manifest"] == manifest:
        if verbose:
            print(f"Maskierte Merkmale '{name}' aus dem Cache geladen ({len(cached['positions'])} Zeilen).")
        return cached
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    rows, coalitions_all = [], []
    features = {key: [] for key in backbones}
    token_players = {key: [] for key in backbones}
    for count, position in enumerate(positions):
        crop = pets.load_crop(ids[int(position)])
        regions = make_regions(crop, "slic", n_players)
        if regions.max() + 1 != n_players:
            msg = f"Image {ids[int(position)]} has {regions.max() + 1} players instead of {n_players}."
            raise ValueError(msg)
        coalitions = sample_coalitions(n_players, per_image, rng)
        images = np.repeat(crop[None], per_image, axis=0)
        for key, backbone in backbones.items():
            tokens_to_players = regions_to_tokens(regions, backbone.patch_size)
            token_players[key].append(torch.from_numpy(tokens_to_players))
            visible = token_coalition_mask(coalitions, tokens_to_players)
            features[key].append(backbone.features(images, visible))
        rows.extend([int(position)] * per_image)
        coalitions_all.append(coalitions)
        if verbose and count % 200 == 0:
            print(f"  {count + 1}/{len(positions)} Bilder, {time.perf_counter() - start:.0f} s")
    payload = {
        "manifest": manifest,
        "positions": torch.tensor(rows),
        "coalitions": torch.from_numpy(np.concatenate(coalitions_all)),
        "features": {key: torch.cat(parts) for key, parts in features.items()},
        "image_positions": torch.as_tensor(np.asarray(positions, dtype=np.int64)),
        "token_players": {key: torch.stack(parts) for key, parts in token_players.items()},
    }
    save(path, payload)
    return payload


def coalition_cache(
    pets: OxfordPets,
    image_id: str,
    backbones: dict,
    *,
    players: str = "slic",
    n_players: int = 16,
    removal: str = "token",
    fill: str = "blur",
) -> CoalitionFeatures:
    """A :class:`~vit_uncertainty.removal.CoalitionFeatures` for one test image, pre-filled from disk.

    shapiq's approximators draw the same coalitions for the same seed, budget, and number of players,
    independent of the game values. So the transformer passes of an explanation can be computed once
    (e.g. by ``scripts/precompute.py``) and every later explanation of the same image, with any probly
    method, only reads them. Call :func:`save_coalition_cache` to write new coalitions back.
    """
    from .players import make_regions  # noqa: PLC0415
    from .removal import CoalitionFeatures  # noqa: PLC0415

    crop = pets.load_crop(image_id)
    regions = make_regions(crop, players, n_players)
    cache = CoalitionFeatures(crop, regions, backbones, removal=removal, fill=fill)
    tag = f"{players}{n_players}_{removal}" + (f"-{fill}" if removal == "fill" else "")
    cache.cache_path = ARTIFACT_DIR / "coalitions" / f"{image_id}_{tag}.pt"
    cache.manifest = {
        "schema": 1,
        "image": image_id,
        "players": players,
        "n_players": n_players,
        "removal": removal,
        "fill": fill if removal == "fill" else None,
        "regions": hashlib.sha256(regions.tobytes()).hexdigest()[:16],
        "models": {key: b.model.pretrained_cfg.get("hf_hub_id", key) for key, b in backbones.items()},
    }
    stored = load(cache.cache_path)
    if stored is not None and stored["manifest"] == cache.manifest:
        cache.load_state(stored)
    return cache


def save_coalition_cache(cache: CoalitionFeatures) -> None:
    """Write all coalitions of a cache from :func:`coalition_cache` back to disk."""
    save(cache.cache_path, {"manifest": cache.manifest, **cache.state()})
