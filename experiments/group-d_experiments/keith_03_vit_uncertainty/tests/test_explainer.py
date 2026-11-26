"""Tests for the probly bridge, the uncertainty game, the explainer, and the learned explainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import shapiq
import torch

from vit_uncertainty.amortized import AmortizedUncertaintyExplainer, efficient, player_tokens
from vit_uncertainty.explainer import UncertaintyExplainer
from vit_uncertainty.game import UncertaintyGame
from vit_uncertainty.ground_truth import coalition_table, moebius_pair
from vit_uncertainty.players import make_regions
from vit_uncertainty.probly_bridge import (
    FeatureLayout,
    build_probly_dropout,
    build_probly_ensemble,
    make_head,
    member,
    uncertainty,
)
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


# ------------------------------------------------------------------ probly construction (since 25.11.)


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.log()).sum(-1)


def test_mixed_ensemble_is_a_probly_ensemble_of_logit_classifiers(backbones, image) -> None:
    """For probly, the mixed ensemble is a ModuleList of logit classifiers; P, TU, AU match a hand computation."""
    from probly.method.ensemble import EnsemblePredictor  # noqa: PLC0415
    from probly.predictor import LogitClassifier  # noqa: PLC0415

    layout = FeatureLayout.from_backbones(backbones)
    members = [
        member(layout, name, make_head(32, 5, seed=seed).state_dict()).double()
        for name, seed in (("vit", 0), ("vit", 1), ("dinov2", 2))
    ]
    ensemble = build_probly_ensemble(members)
    assert isinstance(ensemble, EnsemblePredictor)
    assert all(isinstance(m, LogitClassifier) for m in ensemble)
    features = torch.cat([b.features(image[None]) for b in backbones.values()], dim=1).double()
    scores = uncertainty(ensemble, features)
    with torch.no_grad():
        p = torch.stack([m(features).softmax(-1) for m in members])  # (members, 1, classes)
    mean = p.mean(0)
    np.testing.assert_allclose(scores["probabilities"], mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(scores["TU"], _entropy(mean).numpy(), atol=1e-12)
    np.testing.assert_allclose(scores["AU"], _entropy(p).mean(0).numpy(), atol=1e-12)


def test_dropout_equals_probly_with_predictor_type(backbones, image) -> None:
    """Registering the head ourselves gives exactly what ``predictor_type="logit_classifier"`` gives."""
    from probly.method import dropout  # noqa: PLC0415

    layout = FeatureLayout.from_backbones(backbones)
    state = make_head(32, 5, seed=3).state_dict()
    ours = build_probly_dropout(member(layout, "vit", state).double(), p=0.5)
    # The keyword is added by probly's decorator; passed as a dict, the type checker needs no stubs.
    keyword: dict[str, Any] = {"predictor_type": "logit_classifier"}
    reference = dropout(member(layout, "vit", state).double(), p=0.5, **keyword).eval()
    features = torch.cat([b.features(image[None]) for b in backbones.values()], dim=1).double()
    for key, value in uncertainty(ours, features, num_samples=20).items():
        np.testing.assert_array_equal(value, uncertainty(reference, features, num_samples=20)[key])


# ------------------------------------------------------------------------ indices (since 25.11.)


def test_fsii_reuses_the_coalitions_of_k_sii(methods, backbones, image) -> None:
    """FSII (RegressionFSII) asks for the same coalitions as k-SII: no new transformer passes."""
    regions = make_regions(image, "slic", 16)
    cache = CoalitionFeatures(image, regions, backbones)
    explanations = {
        index: UncertaintyExplainer(
            backbones, methods["Gemischt"], players="slic", n_players=16, index=index, budget=300
        ).explain(image, regions=regions, coalition_features=cache)
        for index in ("k-SII", "FSII")
    }
    fsii = explanations["FSII"]
    assert fsii.n_forward == 0
    assert fsii.values["EU"].index == "FSII"
    np.testing.assert_allclose(fsii.pairs("TU"), fsii.pairs("AU") + fsii.pairs("EU"), atol=1e-6)


def test_exact_indices_agree_for_two_players(methods, backbones, image) -> None:
    """With two players every order-2 index gives the same pair: v(12) - v(1) - v(2) + v(empty)."""
    regions = np.zeros(image.shape[:2], dtype=np.int64)
    regions[:, 112:] = 1
    cache = CoalitionFeatures(image, regions, backbones)
    pairs = {}
    for index in ("k-SII", "STII", "FSII", "Moebius"):
        explainer = UncertaintyExplainer(backbones, methods["Gemischt"], index=index, max_order=2, budget=4)
        pairs[index] = explainer.explain(image, regions=regions, coalition_features=cache).pairs("EU")[0, 1]
    table = coalition_table(cache, methods["Gemischt"], target_class=0)
    for value in pairs.values():
        assert value == pytest.approx(moebius_pair(table, "EU", 0, 1), abs=1e-12)


def test_exact_indices_share_the_three_way_term_differently(methods, backbones, image) -> None:
    """With three players the indices differ only in how they share the three-way Moebius term m(012).

    Pair (0, 1): Moebius keeps m(01); k-SII and FSII add half of m(012), STII a third. All stay linear,
    so pair(TU) = pair(AU) + pair(EU) for every index.
    """
    regions = make_regions(image, "grid", 4)
    regions[regions == 3] = 2  # three players
    cache = CoalitionFeatures(image, regions, backbones)
    explanations = {
        index: UncertaintyExplainer(backbones, methods["Gemischt"], index=index, max_order=2, budget=8).explain(
            image, regions=regions, coalition_features=cache
        )
        for index in ("k-SII", "STII", "FSII", "Moebius")
    }
    moebius = explanations["Moebius"].values["EU"]
    pair, three_way = moebius[(0, 1)], moebius[(0, 1, 2)]
    share = {"k-SII": 1 / 2, "FSII": 1 / 2, "STII": 1 / 3, "Moebius": 0.0}
    for index, explanation in explanations.items():
        assert explanation.pairs("EU")[0, 1] == pytest.approx(pair + share[index] * three_way, abs=1e-12)
        np.testing.assert_allclose(
            explanation.pairs("TU"), explanation.pairs("AU") + explanation.pairs("EU"), atol=1e-10
        )
    assert cache.n_forward == 8


def test_approximated_stii_is_refused(methods, backbones, image) -> None:
    """STII has no regression approximator here; with too small a budget the explainer says so."""
    explainer = UncertaintyExplainer(
        backbones, methods["Gemischt"], players="grid", n_players=9, index="STII", budget=64
    )
    with pytest.raises(ValueError, match="No approximator"):
        explainer.explain(image)
