"""Frozen vision transformers that can hide patch tokens.

Why two transformers? In a pilot study on skin images (not in this repository, see the README) heads on
one frozen network agreed almost always, so their epistemic uncertainty (EU) was close to zero. Members that look at
the image *differently* disagree more. We therefore use two small transformers that were trained
in very different ways:

* ``vit``: ViT-S/16, trained **supervised** on ImageNet-21k and fine-tuned on ImageNet-1k,
* ``dinov2``: DINOv2-S/14, trained **self-supervised** on 142 million images without labels.

Both are used as frozen feature extractors: the output is the class token (CLS) after the final
layer norm, a 384-dimensional vector per image.

How a player is removed: in token space, not in pixel space. A hidden patch token is excluded
from attention (no other token may look at it). For the CLS output this is exactly the same as
deleting the token from the sequence (the ``"remove"`` strategy of ``shapiq_games``), because the
layer norms and MLPs act on every token separately. The position embeddings of the visible tokens
stay where they are. ``tests/test_backbones.py`` checks this equivalence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from timm.models.vision_transformer import VisionTransformer


class BackboneSpec(TypedDict):
    """How to load one backbone with timm, and how to call it in tables and plots."""

    timm_name: str
    kwargs: dict[str, Any]  # extra arguments of timm.create_model, e.g. the image size
    label: str
    description: str


BACKBONES: dict[str, BackboneSpec] = {
    "vit": {
        "timm_name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "kwargs": {},
        "label": "ViT-S/16",
        "description": "supervised on ImageNet-21k, fine-tuned on ImageNet-1k",
    },
    "dinov2": {
        "timm_name": "vit_small_patch14_dinov2.lvd142m",
        "kwargs": {"img_size": 224},  # the checkpoint is trained at 518 px; timm resizes the position embeddings
        "label": "DINOv2-S/14",
        "description": "self-supervised (DINOv2) on LVD-142M",
    },
}


class FrozenViT(nn.Module):
    """A frozen timm vision transformer returning CLS features, optionally with hidden patch tokens.

    Args:
        name: Key of :data:`BACKBONES`, or any label when ``model`` is given.
        model: Optional ready timm ``VisionTransformer`` (used by the tests with a tiny random model).
        mean: Normalization mean; defaults to the checkpoint's own value.
        std: Normalization standard deviation; defaults to the checkpoint's own value.
    """

    def __init__(
        self,
        name: str,
        *,
        model: nn.Module | None = None,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ) -> None:
        """Load (or wrap) the transformer and read its patch geometry."""
        super().__init__()
        # timm is imported here (not at the top) so that importing the package stays fast.
        import timm  # noqa: PLC0415
        from timm.models.vision_transformer import VisionTransformer  # noqa: PLC0415

        if model is None:
            spec = BACKBONES[name]
            # num_classes=0 removes the ImageNet classifier; the model then returns the CLS features.
            model = timm.create_model(spec["timm_name"], pretrained=True, num_classes=0, **spec["kwargs"])
        # Hiding tokens needs the attention mask of timm's VisionTransformer (both checkpoints are one).
        # The check also tells the type checker which attributes the model has.
        if not isinstance(model, VisionTransformer):
            msg = f"FrozenViT needs a timm VisionTransformer, got {type(model).__name__}."
            raise TypeError(msg)
        self.name = name
        self.model: VisionTransformer = model.requires_grad_(False).eval()
        config = getattr(model, "pretrained_cfg", {}) or {}
        self.mean = tuple(mean or config.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(std or config.get("std", (0.229, 0.224, 0.225)))
        patch_embed = model.patch_embed  # timm's PatchEmbed; its type hint in timm is a plain Callable
        self.patch_size = int(patch_embed.patch_size[0])
        self.grid_size = tuple(int(v) for v in patch_embed.grid_size)
        self.n_patches = self.grid_size[0] * self.grid_size[1]
        self.n_prefix = int(model.num_prefix_tokens)  # CLS (and register tokens, if any)
        self.feature_dim = int(model.num_features)

    @property
    def label(self) -> str:
        """Readable model name for tables and plots."""
        spec = BACKBONES.get(self.name)
        return self.name if spec is None else spec["label"]

    def normalize(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert ``uint8`` images ``(N, H, W, 3)`` to normalized float tensors ``(N, 3, H, W)``."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(np.ascontiguousarray(images))
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            msg = "Expected uint8 images with shape (N, H, W, 3)."
            raise ValueError(msg)
        tensor = images.permute(0, 3, 1, 2).float() / 255
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _attention_mask(self, token_visible: torch.Tensor) -> torch.Tensor:
        # Prefix tokens (CLS) are always visible. Shape (N, 1, 1, T): the same keys are hidden for
        # every head and every query; True means "may be attended to".
        prefix = torch.ones(token_visible.shape[0], self.n_prefix, dtype=torch.bool)
        keys = torch.cat([prefix, token_visible.bool()], dim=1)
        return keys[:, None, None, :]

    @torch.no_grad()
    def forward(self, images: torch.Tensor, token_visible: torch.Tensor | None = None) -> torch.Tensor:
        """CLS features of normalized images; hidden patch tokens are excluded from attention.

        Args:
            images: Normalized float tensor ``(N, 3, H, W)``.
            token_visible: Optional boolean tensor ``(N, n_patches)``; ``True`` = token visible.

        Returns:
            Float tensor ``(N, feature_dim)``.
        """
        mask = None if token_visible is None else self._attention_mask(token_visible)
        tokens = self.model.forward_features(images, attn_mask=mask)
        return self.model.forward_head(tokens, pre_logits=True)

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Final (layer-normed) patch tokens of the full images, shape ``(N, n_patches, feature_dim)``.

        Used as input of the learned explainer (ViT Shapley): it reads these tokens and predicts
        one value per token.
        """
        tokens = self.model.forward_features(images)
        return tokens[:, self.n_prefix :]

    def features(
        self,
        images: np.ndarray,
        token_visible: np.ndarray | None = None,
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """CLS features of ``uint8`` images in bounded batches (convenience wrapper around :meth:`forward`)."""
        outputs = []
        for start, stop in _batches(len(images), batch_size):
            visible = None if token_visible is None else torch.as_tensor(token_visible[start:stop])
            outputs.append(self(self.normalize(images[start:stop]), visible).float())
        features = torch.cat(outputs)
        if not torch.isfinite(features).all():
            msg = f"{self.name} produced non-finite features."
            raise ValueError(msg)
        return features


def load_backbones(names: tuple[str, ...] = ("vit", "dinov2")) -> dict[str, FrozenViT]:
    """Load the frozen transformers by name (downloads the weights from Hugging Face once)."""
    return {name: FrozenViT(name) for name in names}


def _batches(n: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(start + size, n)
