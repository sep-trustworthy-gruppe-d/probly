"""Tests for players, token assignment, and token removal."""

from __future__ import annotations

import numpy as np
import torch

from vit_uncertainty.players import grid_regions, make_regions, regions_to_tokens, slic_regions, token_coalition_mask


def test_grid_regions_cover_all_players_equally() -> None:
    regions = grid_regions(4, 4)
    assert regions.shape == (224, 224)
    assert set(np.unique(regions)) == set(range(16))
    assert len(set(np.bincount(regions.ravel()))) == 1  # 224 / 4 = 56: all cells equal


def test_slic_returns_exactly_the_requested_number(image) -> None:
    regions = slic_regions(image, n_segments=16)
    assert regions.max() + 1 == 16
    assert set(np.unique(regions)) == set(range(16))


def test_slic_on_a_uniform_image_still_gives_the_requested_number() -> None:
    """SLIC finds almost nothing in a flat image; the fallback splits regions until there are 16."""
    flat = np.full((224, 224, 3), 128, dtype=np.uint8)
    regions = slic_regions(flat, n_segments=16)
    assert set(np.unique(regions)) == set(range(16))


def test_grid_tokens_follow_the_grid() -> None:
    # A 2 x 2 grid on 14 x 14 tokens of 16 px: the upper-left 7 x 7 tokens belong to player 0.
    tokens = regions_to_tokens(grid_regions(2, 2), patch_size=16).reshape(14, 14)
    assert (tokens[:7, :7] == 0).all()
    assert (tokens[7:, 7:] == 3).all()


def test_token_mask_hides_exactly_the_absent_players() -> None:
    token_players = np.array([0, 0, 1, 2, 2])
    mask = token_coalition_mask(np.array([[True, False, True]]), token_players)
    assert mask.tolist() == [[True, True, False, True, True]]


def test_all_visible_mask_equals_plain_forward(backbones, image) -> None:
    backbone = backbones["vit"]
    plain = backbone.features(image[None])
    masked = backbone.features(image[None], np.ones((1, backbone.n_patches), dtype=bool))
    torch.testing.assert_close(plain, masked, rtol=0, atol=1e-6)


def test_hiding_tokens_equals_deleting_them(backbones, image) -> None:
    """Attention masking is the same as removing the tokens from the sequence (for the CLS output)."""
    for backbone in backbones.values():
        regions = make_regions(image, "grid", 9)
        coalition = np.array([[True, False, True, False, True, False, True, False, True]])
        visible = token_coalition_mask(coalition, regions_to_tokens(regions, backbone.patch_size))[0]
        masked = backbone.features(image[None], visible[None])
        model = backbone.model
        with torch.no_grad():
            tokens = model.norm_pre(model._pos_embed(model.patch_embed(backbone.normalize(image[None]))))
            kept_patches = backbone.n_prefix + torch.from_numpy(np.flatnonzero(visible))
            keep = torch.cat([torch.arange(backbone.n_prefix), kept_patches])
            out = model.norm(model.blocks(tokens[:, keep]))
            deleted = model.forward_head(out, pre_logits=True)
        torch.testing.assert_close(masked, deleted, rtol=1e-4, atol=1e-4)


def test_hidden_content_does_not_matter(backbones, image) -> None:
    """Changing the pixels of a hidden player must not change the output."""
    backbone = backbones["vit"]
    regions = make_regions(image, "grid", 4)
    coalition = np.array([[True, True, True, False]])
    visible = token_coalition_mask(coalition, regions_to_tokens(regions, backbone.patch_size))
    changed = image.copy()
    changed[regions == 3] = 255 - changed[regions == 3]
    torch.testing.assert_close(backbone.features(image[None], visible), backbone.features(changed[None], visible))
