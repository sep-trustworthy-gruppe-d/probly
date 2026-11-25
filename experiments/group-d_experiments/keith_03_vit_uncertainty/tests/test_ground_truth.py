"""Tests for the ground-truth test images (notebook 03-3), with tiny random transformers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from vit_uncertainty import CoalitionFeatures, OxfordPets, UncertaintyExplainer, ground_truth as gt
from vit_uncertainty.data import CROP_SIZE


def test_checkerboard_alternates() -> None:
    board = gt.checkerboard(48, 8)
    assert board.shape == (48, 48, 3)
    assert board.dtype == np.uint8
    assert set(np.unique(board)) == {0, 255}
    # Neighbouring squares differ, squares are 8 px wide.
    assert board[0, 0, 0] != board[0, 8, 0]
    assert board[0, 0, 0] == board[0, 7, 0] == board[8, 8, 0]


def test_background_positions_respect_trimap() -> None:
    trimap = np.full((224, 224), 2)
    trimap[:, 112:] = 1  # right half is pet
    positions = gt.background_positions(trimap)
    assert positions
    for row, col in positions:
        assert row % 16 == 0
        assert col % 16 == 0
        assert col + gt.PATCH_SIZE <= 112 + 0.1 * gt.PATCH_SIZE  # at most 10 % of the patch on the pet


def test_pattern_image_players(image: np.ndarray) -> None:
    trimap = np.full((224, 224), 2)
    trimap[80:200, 80:200] = 1
    patch = gt.checkerboard()
    test = gt.make_pattern_image(image, trimap, patch, (16, 16), variant="test")
    assert np.array_equal(test.image[16:64, 16:64], patch)
    assert np.array_equal(test.image[100:, 100:], image[100:, 100:])  # nothing else changed
    assert (test.players[test.cause] == 0).all()
    assert test.cause.sum() == 48 * 48
    assert (test.players[100:150, 100:150] == 1).all()  # pet
    assert test.players[220, 5] == 2  # background
    assert test.player_names == gt.PATTERN_PLAYERS


def test_two_breeds_image_halves(image: np.ndarray) -> None:
    other = image[::-1].copy()
    test = gt.make_two_breeds_image(image, other, variant="test")
    a, b = gt.STRIP
    assert test.image.shape == image.shape
    assert np.array_equal(test.image[:, :112], image[:, a:b])
    assert np.array_equal(test.image[:, 112:], other[:, a:b])
    assert (test.players[:, :112] == 0).all()
    assert (test.players[:, 112:] == 1).all()


def test_chimera_image_pastes_head(image: np.ndarray) -> None:
    body = np.zeros_like(image)
    trimap = np.full((224, 224), 2)
    trimap[60:220, 40:180] = 1
    test = gt.make_chimera_image(image, (0, 0, 56, 56), body, (40, 60, 120, 140), trimap, variant="test")
    assert test.image[60:140, 40:120].any()  # the head arrived (the body image is black)
    assert not test.image[150:, :].any()  # outside the box the body image is unchanged
    assert (test.players[60:140, 40:120] == 0).all()
    assert (test.players[150:200, 60:160] == 1).all()
    assert test.players[5, 5] == 2


def test_all_coalitions_order() -> None:
    coalitions = gt.all_coalitions(3)
    assert coalitions.shape == (8, 3)
    assert not coalitions[0].any()
    assert coalitions[-1].all()
    assert coalitions[1].tolist() == [True, False, False]  # player 0 is the lowest bit


def test_exact_pair_equals_moebius(backbones, methods, image) -> None:
    # With two coarse players and budget >= 4, shapiq computes exact values. The k-SII pair value
    # must then be the pure interaction v(LR) - v(L) - v(R) + v(empty) from the coalition table.
    test = gt.make_two_breeds_image(image, image[::-1].copy(), variant="test")
    predictor = methods["Gemischt"]
    explainer = UncertaintyExplainer(backbones, predictor, budget=4, random_state=0)
    # explain_function instead of explain: shapiq types explain() as returning InteractionValues, ours returns an
    # UncertaintyExplanation (known design point, see the handover, task S2). The result is the same.
    explanation = explainer.explain_function(test.image, regions=test.players)
    features = CoalitionFeatures(test.image, test.players, backbones)
    table = gt.coalition_table(features, predictor, target_class=explanation.target_class)
    for quantity in ("P", "TU", "AU", "EU"):
        assert explanation.pairs(quantity)[0, 1] == pytest.approx(gt.moebius_pair(table, quantity, 0, 1), abs=1e-9)
        # Efficiency of the exact Shapley values.
        total = explanation.first_order(quantity).sum()
        assert total == pytest.approx(gt.value_of(table, quantity, (0, 1)) - gt.value_of(table, quantity, ()), abs=1e-9)
    # The decomposition carries over to the pairs: pair(TU) = pair(AU) + pair(EU).
    assert explanation.pairs("TU")[0, 1] == pytest.approx(
        explanation.pairs("AU")[0, 1] + explanation.pairs("EU")[0, 1], abs=1e-9
    )


def test_players_have_tokens(backbones) -> None:
    players = np.zeros((224, 224), dtype=np.int64)
    players[:, 112:] = 1
    assert gt.players_have_tokens(players, backbones)
    players[0:2, 0:2] = 2  # a tiny third player owns no token
    assert not gt.players_have_tokens(players, backbones)


def test_cause_superpixel_and_rank() -> None:
    regions = np.zeros((4, 4), dtype=np.int64)
    regions[:, 2:] = 1
    cause = np.zeros((4, 4), dtype=bool)
    cause[:, 3] = True
    assert gt.cause_superpixel(regions, cause) == 1
    values = np.array([0.1, 0.5, -0.2])
    assert gt.rank_of(values, 1) == 1
    assert gt.rank_of(values, 0) == 2
    assert gt.rank_of(values, 2) == 3


def test_stored_features_round_trip(backbones, image, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gt, "ARTIFACT_DIR", tmp_path)
    players = np.zeros((224, 224), dtype=np.int64)
    players[:, 112:] = 1
    first = gt.StoredCoalitionFeatures(image, players, backbones, "probe")
    values = first(gt.all_coalitions(2))
    first.store()
    second = gt.StoredCoalitionFeatures(image, players, backbones, "probe")
    again = second(gt.all_coalitions(2))
    assert second.n_forward == 0  # everything came from disk
    assert np.allclose(values.numpy(), again.numpy())


def test_load_head_box_moves_box_into_crop(tmp_path: Path) -> None:
    # A fake dataset with one 400 x 300 image: resize(256) gives 341 x 256, the crop starts at x = 58.
    folder = tmp_path / "oxford-iiit-pet"
    (folder / "images").mkdir(parents=True)
    (folder / "annotations" / "xmls").mkdir(parents=True)
    Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)).save(folder / "images" / "cat_1.jpg")
    xml = (
        "<annotation><object><bndbox><xmin>150</xmin><ymin>60</ymin><xmax>250</xmax><ymax>180</ymax>"
        "</bndbox></object></annotation>"
    )
    (folder / "annotations" / "xmls" / "cat_1.xml").write_text(xml)
    pets = OxfordPets(tmp_path)
    box = pets.load_head_box("cat_1")
    scale = 256 / 300
    left = round((int(256 * 400 / 300) - CROP_SIZE) / 2)
    assert box == (
        round(150 * scale - left),
        round(60 * scale - 16),
        round(250 * scale - left),
        round(180 * scale - 16),
    )
    assert pets.load_head_box("missing_1") is None
    # A box mostly outside the crop is rejected.
    (folder / "annotations" / "xmls" / "cat_1.xml").write_text(xml.replace(">150<", ">0<").replace(">250<", ">60<"))
    assert pets.load_head_box("cat_1") is None
