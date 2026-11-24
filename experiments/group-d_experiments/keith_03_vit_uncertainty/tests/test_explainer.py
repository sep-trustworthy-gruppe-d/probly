"""Tests for the probly bridge, the uncertainty game, the explainer, and the learned explainer."""

from __future__ import annotations

import numpy as np
import pytest
import shapiq
import torch

from vit_uncertainty.amortized import AmortizedUncertaintyExplainer, efficient, player_tokens
from vit_uncertainty.explainer import UncertaintyExplainer
from vit_uncertainty.game import UncertaintyGame
from vit_uncertainty.players import make_regions
from vit_uncertainty.probly_bridge import uncertainty
from vit_uncertainty.removal import CoalitionFeatures


def test_decomposition_and_probabilities(methods, backbones, image) -> None:
    features = torch.cat([b.features(image[None]) for b in backbones.values()], dim=1)
    for predictor in methods.values():
        scores = uncertainty(predictor, features)
        np.testing.assert_allclose(scores["TU"], scores["AU"] + scores["EU"], atol=1e-10)
        np.testing.assert_allclose(scores["probabilities"].sum(1), 1.0, atol=1e-10)
        assert (scores["EU"] >= -1e-12).all()


def test_mc_dropout_values_are_deterministic(methods, backbones, image) -> None:
    """The same coalition gets the same value, alone or together with others (fixed dropout masks)."""
    regions = make_regions(image, "grid", 4)
    cache = CoalitionFeatures(image, regions, backbones)
    coalitions = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=bool)
    game = UncertaintyGame(cache, methods["MC Dropout ViT"], "EU")
    together = game.value_function(coalitions)
    alone = UncertaintyGame(cache, methods["MC Dropout ViT"], "EU").value_function(coalitions[:1])
    np.testing.assert_allclose(together[0], alone[0], atol=1e-12)


def test_cache_evaluates_every_coalition_once(backbones, image) -> None:
    cache = CoalitionFeatures(image, make_regions(image, "grid", 4), backbones)
    coalitions = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
    first = cache(coalitions)
    cache(coalitions)
    assert cache.n_forward == 2
    torch.testing.assert_close(first[0], first[1])


@pytest.mark.parametrize("method", ["Ensemble ViT", "MC Dropout ViT", "Gemischt"])
def test_exact_explanation_properties(methods, backbones, image, method) -> None:
    """Efficiency for every quantity and phi(TU) = phi(AU) + phi(EU) with exact Shapley values."""
    explainer = UncertaintyExplainer(
        backbones, methods[method], players="grid", n_players=9, index="SV", max_order=1, budget=2**9
    )
    explanation = explainer.explain(image)
    assert isinstance(explainer, shapiq.Explainer)
    for quantity in ("P", "TU", "AU", "EU"):
        total = explanation.empty[quantity] + explanation.first_order(quantity).sum()
        assert total == pytest.approx(explanation.full[quantity], abs=1e-9)
    np.testing.assert_allclose(
        explanation.first_order("TU"), explanation.first_order("AU") + explanation.first_order("EU"), atol=1e-9
    )
    assert explanation.n_forward == 2**9  # four games, one forward pass per coalition


def test_dummy_player_gets_zero(methods, backbones, image) -> None:
    """A player without tokens cannot change anything, so its Shapley value is zero."""
    regions = make_regions(image, "grid", 4)
    regions[0, 0] = 4  # player 4 owns a single pixel: no token is assigned to it
    explanation = UncertaintyExplainer(backbones, methods["Gemischt"], index="SV", max_order=1, budget=2**5).explain(
        image, regions=regions
    )
    for quantity in ("P", "TU", "AU", "EU"):
        assert explanation.first_order(quantity)[4] == pytest.approx(0.0, abs=1e-10)


def test_shapley_values_come_for_free_and_are_efficient(methods, backbones, image) -> None:
    """With k-SII as index, the explainer also returns Shapley values from the same coalitions."""
    explainer = UncertaintyExplainer(backbones, methods["Gemischt"], players="slic", n_players=16, budget=300)
    explanation = explainer.explain(image)
    assert explanation.n_forward <= 300  # k-SII and SV for four quantities share the coalitions
    for quantity in ("P", "TU", "AU", "EU"):
        total = explanation.empty[quantity] + explanation.first_order(quantity).sum()
        assert total == pytest.approx(explanation.full[quantity], abs=1e-6)
    np.testing.assert_allclose(
        explanation.first_order("TU"), explanation.first_order("AU") + explanation.first_order("EU"), atol=1e-6
    )


def test_approximation_keeps_decomposition_and_shares_coalitions(methods, backbones, image) -> None:
    """The decomposition carries over to the pairs when the budget exceeds the number of unknowns.

    k-SII up to order 2 with 16 players has 1 + 16 + 120 = 137 unknowns; with a larger budget the
    approximation is linear in the game.
    """
    explainer = UncertaintyExplainer(backbones, methods["Gemischt"], players="slic", n_players=16, budget=300)
    explanation = explainer.explain(image)
    np.testing.assert_allclose(explanation.pairs("TU"), explanation.pairs("AU") + explanation.pairs("EU"), atol=1e-6)
    assert explanation.n_forward <= 300


def test_normalization_makes_values_efficient() -> None:
    torch.manual_seed(0)
    phi, gap = torch.randn(2, 3, 4), torch.randn(2, 4)
    torch.testing.assert_close(efficient(phi, gap).sum(1), gap)


def test_player_tokens_are_means_and_area_shares() -> None:
    tokens = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)  # 4 tokens with 2 features
    players = player_tokens(tokens, torch.tensor([[0, 0, 1, 1]]), 2)
    torch.testing.assert_close(players[0, :, :2], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    torch.testing.assert_close(players[0, :, 2], torch.tensor([0.5, 0.5]))


def test_amortized_explainer_recovers_an_additive_game() -> None:
    """For an additive game v(S) = sum_{i in S} w_i, the Shapley values are the w_i.

    Here w_i is the mean of the first feature of the tokens of player i, something the explainer can
    represent from its player tokens. Trained only on sampled coalitions (no Shapley values given),
    it must find these weights.
    """
    torch.manual_seed(0)
    n_images, n_tokens, n_players = 128, 12, 4
    tokens = torch.randn(n_images, n_tokens, 8)
    token_players = torch.arange(n_tokens).remainder(n_players).repeat(n_images, 1)
    weights = torch.stack([tokens[:, token_players[0] == p, 0].mean(1) for p in range(n_players)], dim=1)
    weights = weights[..., None].repeat(1, 1, 4)
    rng = np.random.default_rng(0)
    coalitions = torch.from_numpy(rng.integers(0, 2, size=(n_images * 8, n_players)).astype(bool))
    rows = torch.arange(n_images).repeat_interleave(8)
    data = {
        "tokens": tokens,
        "token_players": token_players,
        "empty": torch.zeros(4),
        "full": weights.sum(1),
        "rows": rows,
        "coalitions": coalitions,
        "values": torch.einsum("mn,mnq->mq", coalitions.float(), weights[rows]),
    }
    explainer = AmortizedUncertaintyExplainer(n_players, token_dim=8, width=32, depth=1)
    explainer.fit(data, data, epochs=300, lr=3e-3, verbose=False)
    phi = explainer.explain(tokens, token_players, data["full"], data["empty"])
    error = np.abs(phi - weights.numpy()).mean() / np.abs(weights.numpy()).mean()
    assert error < 0.1
