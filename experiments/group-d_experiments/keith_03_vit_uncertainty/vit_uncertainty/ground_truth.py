"""Ground truth for image explanations: test images whose cause of uncertainty we build in ourselves.

Real images have no ground truth: nobody knows which superpixel "really" makes the model uncertain.
So we build test images where we *know* where the cause sits, like the synthetic tables of the
earlier tabular work. Three constructions, each with a control that differs only in the cause:

=============  =================================================  ==============================================
test           what we build                                      control (same construction, no cause)
=============  =================================================  ==============================================
pattern        a 48 x 48 checkerboard patch on the background     a natural background patch from another image
two breeds     left half breed A, right half breed B              two images of the **same** breed
chimera        head of breed A pasted onto the body of breed B    head of another image of breed B (same paste)
=============  =================================================  ==============================================

For every test image we also build **coarse players**: the parts the image was made of (patch / pet /
background, left / right, head / body / background). With 2 or 3 players shapiq evaluates all 4 or 8
coalitions and the values are **exact**. So the ground truth is not blurred by an approximation.

Two levels are checked separately (see notebook 03-3):

1. Does the **model** react to the cause at all? (compare game values with and without it)
2. Does the **explanation** find it? (Shapley values and pair interactions on the known parts)

What is known and what is only expected: we know *where* the cause is and whether it needs *two parts
together*. Whether it shows up as AU or as EU is a hypothesis that the tests check, not a known fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .features import ARTIFACT_DIR, load, save
from .players import regions_to_tokens
from .probly_bridge import uncertainty
from .removal import CoalitionFeatures

if TYPE_CHECKING:
    from torch import nn

CROP_SIZE = 224

# Pattern test. 48 pixels = 3 x 3 ViT tokens (16 px); the position is a multiple of 16, so the patch
# covers whole ViT tokens. It is about 4.6 % of the image: small enough that the pet stays fully
# visible, large enough to be a few tokens for both transformers (DINOv2 tokens are 14 px).
PATCH_SIZE = 48
PATCH_STRIDE = 16
CHECKER_SQUARE = 8  # 8 px black and white squares: a sharp, regular texture that no pet photo contains
MIN_BACKGROUND = 0.9  # the patch goes where at least 90 % of the pixels are background (trimap)

# Two-breeds test: the central vertical strip of each crop (columns 56..168) fills one half. Pets are
# usually centered, so the strip shows most of the pet. 112 px = 7 ViT tokens = 8 DINOv2 tokens, so the
# border between the halves lies exactly on a token border for both transformers.
STRIP = (56, 168)

# Chimera test: head boxes from annotations/xmls must have a reasonable size in the crop.
MIN_HEAD_SIDE = 40
MAX_HEAD_SIDE = 170
# ... and the body (pet pixels outside the head box) must cover at least 10 % of the image. Some photos
# show almost only the head (close-ups); a "head x body" pair would be meaningless there.
MIN_BODY_SHARE = 0.1

PATTERN_PLAYERS = ("Flicken", "Tier", "Hintergrund")
TWO_BREEDS_PLAYERS = ("links", "rechts")
CHIMERA_PLAYERS = ("Kopf", "Körper", "Hintergrund")


@dataclass
class TestImage:
    """One constructed test image with its coarse players and its known cause.

    Attributes:
        kind: ``"muster"``, ``"zwei_rassen"`` or ``"chimaere"``.
        variant: ``"test"`` (with the cause) or ``"kontrolle"`` (same construction without the cause).
        image: ``uint8`` RGB image ``(224, 224, 3)``.
        players: Coarse player id per pixel ``(224, 224)``.
        player_names: Name of every coarse player.
        cause: Boolean mask of the pixels that carry the planted cause (patch or pasted head).
        sources: Ids and breeds of the source images.
    """

    kind: str
    variant: str
    image: np.ndarray
    players: np.ndarray
    player_names: tuple[str, ...]
    cause: np.ndarray
    sources: dict = field(default_factory=dict)

    @property
    def n_players(self) -> int:
        """Number of coarse players."""
        return len(self.player_names)


# ---------------------------------------------------------------------------- building test images


def checkerboard(size: int = PATCH_SIZE, square: int = CHECKER_SQUARE) -> np.ndarray:
    """A black-and-white checkerboard ``(size, size, 3)`` with ``square`` pixel squares."""
    index = np.arange(size) // square
    board = ((index[:, None] + index[None, :]) % 2 * 255).astype(np.uint8)
    return np.repeat(board[..., None], 3, axis=2)


def background_positions(
    trimap: np.ndarray, size: int = PATCH_SIZE, stride: int = PATCH_STRIDE, min_background: float = MIN_BACKGROUND
) -> list[tuple[int, int]]:
    """Top-left corners ``(row, col)`` on a ``stride`` grid where a patch lies on the background.

    Args:
        trimap: Oxford Pets trimap aligned with the crop (1 = pet, 2 = background, 3 = border).
        size: Side length of the patch.
        stride: Step of the grid of candidate positions.
        min_background: Smallest share of background pixels under the patch.

    Returns:
        All qualifying positions, row by row.
    """
    background = trimap == 2
    return [
        (row, col)
        for row in range(0, trimap.shape[0] - size + 1, stride)
        for col in range(0, trimap.shape[1] - size + 1, stride)
        if background[row : row + size, col : col + size].mean() >= min_background
    ]


def make_pattern_image(
    base: np.ndarray,
    trimap: np.ndarray,
    patch: np.ndarray,
    position: tuple[int, int],
    *,
    variant: str,
    sources: dict | None = None,
) -> TestImage:
    """Paste ``patch`` onto ``base`` at ``position``; players are patch, pet and background.

    The pet player is the trimap's pet plus border outside the patch (as in notebook 03-1, the border
    counts as pet); everything else is background.
    """
    row, col = position
    size = patch.shape[0]
    image = base.copy()
    image[row : row + size, col : col + size] = patch
    cause = np.zeros(base.shape[:2], dtype=bool)
    cause[row : row + size, col : col + size] = True
    players = np.where(trimap != 2, 1, 2).astype(np.int64)
    players[cause] = 0
    return TestImage("muster", variant, image, players, PATTERN_PLAYERS, cause, sources or {})


def make_two_breeds_image(
    left: np.ndarray, right: np.ndarray, *, variant: str, sources: dict | None = None
) -> TestImage:
    """Left half: central strip of ``left``; right half: central strip of ``right``.

    There is no single "cause" region: the expected effect needs **both** halves together (the model
    sees two breeds at once). ``cause`` therefore marks the whole image.
    """
    a, b = STRIP
    width = b - a
    image = np.concatenate([left[:, a:b], right[:, a:b]], axis=1)
    players = np.zeros(image.shape[:2], dtype=np.int64)
    players[:, width:] = 1
    cause = np.ones(image.shape[:2], dtype=bool)
    return TestImage("zwei_rassen", variant, image, players, TWO_BREEDS_PLAYERS, cause, sources or {})


def make_chimera_image(
    head_image: np.ndarray,
    head_box: tuple[int, int, int, int],
    body_image: np.ndarray,
    body_box: tuple[int, int, int, int],
    body_trimap: np.ndarray,
    *,
    variant: str,
    sources: dict | None = None,
) -> TestImage:
    """Paste the head box of ``head_image`` (resized) into the head box of ``body_image``.

    Players: the pasted box ("Kopf"), the pet pixels of the body image outside the box ("Körper"), and
    the rest ("Hintergrund"). The box also carries a little background of the head image. The control
    pastes a head of the *same* breed in the same way, so paste edges and box background are present in
    test and control alike; only the impossible combination differs.

    Args:
        head_image: Crop that provides the head.
        head_box: ``(x0, y0, x1, y1)`` of the head in ``head_image`` (crop pixels).
        body_image: Crop that provides body and background.
        body_box: ``(x0, y0, x1, y1)`` of the original head in ``body_image``; the new head goes there.
        body_trimap: Trimap of ``body_image``.
        variant: ``"test"`` or ``"kontrolle"``.
        sources: Ids and breeds, stored in the result.
    """
    hx0, hy0, hx1, hy1 = head_box
    bx0, by0, bx1, by1 = body_box
    head = Image.fromarray(np.ascontiguousarray(head_image[hy0:hy1, hx0:hx1]))
    head = np.asarray(head.resize((bx1 - bx0, by1 - by0), Image.Resampling.BILINEAR))
    image = body_image.copy()
    image[by0:by1, bx0:bx1] = head
    cause = np.zeros(image.shape[:2], dtype=bool)
    cause[by0:by1, bx0:bx1] = True
    players = np.where(body_trimap != 2, 1, 2).astype(np.int64)
    players[cause] = 0
    return TestImage("chimaere", variant, image, players, CHIMERA_PLAYERS, cause, sources or {})


def head_box_usable(box: tuple[int, int, int, int] | None) -> bool:
    """``True`` if a head box exists and has a reasonable size in the crop."""
    if box is None:
        return False
    width, height = box[2] - box[0], box[3] - box[1]
    return min(width, height) >= MIN_HEAD_SIDE and max(width, height) <= MAX_HEAD_SIDE


def players_have_tokens(players: np.ndarray, backbones: dict) -> bool:
    """``True`` if every coarse player owns at least one patch token in every backbone.

    A player without tokens could never be hidden in token space; it would be a dummy by construction
    and its value would be meaningless. Such test images are skipped.
    """
    n_players = int(players.max()) + 1
    for backbone in backbones.values():
        owners = regions_to_tokens(players, backbone.patch_size)
        if len(np.unique(owners)) < n_players:
            return False
    return True


# ------------------------------------------------------------------------------- evaluating games


def all_coalitions(n_players: int) -> np.ndarray:
    """All ``2**n`` coalitions as a boolean matrix, ordered like binary numbers (empty set first)."""
    rows = np.array(list(product([False, True], repeat=n_players)), dtype=bool)
    return rows[:, ::-1]  # player 0 = lowest bit, as in removal.coalition_codes


def coalition_table(
    features: CoalitionFeatures,
    predictor: nn.Module,
    *,
    target_class: int,
    num_samples: int = 50,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Game values of **every** coalition of a small game: P, TU, AU, EU and the probabilities.

    With 2 or 3 coarse players this is the complete game (4 or 8 rows). It shows directly what shapiq
    then summarizes, e.g. "EU with head and body visible" against "EU with only the head visible".

    Returns:
        ``{"coalitions": (2**n, n) bool, "P", "TU", "AU", "EU": (2**n,), "probabilities": (2**n, C)}``.
    """
    coalitions = all_coalitions(features.n_players)
    scores = uncertainty(predictor, features(coalitions), num_samples=num_samples, seed=seed)
    return {
        "coalitions": coalitions,
        "P": scores["probabilities"][:, target_class],
        "TU": scores["TU"],
        "AU": scores["AU"],
        "EU": scores["EU"],
        "probabilities": scores["probabilities"],
    }


def value_of(table: dict, quantity: str, visible: tuple[int, ...]) -> float:
    """Game value of the coalition in which exactly the players ``visible`` are visible."""
    wanted = np.zeros(table["coalitions"].shape[1], dtype=bool)
    wanted[list(visible)] = True
    row = int(np.flatnonzero((table["coalitions"] == wanted).all(axis=1))[0])
    return float(table[quantity][row])


def moebius_pair(table: dict, quantity: str, i: int, j: int) -> float:
    """Pure interaction of players ``i`` and ``j`` with all other players hidden.

    ``v({i, j}) - v({i}) - v({j}) + v(empty)``: how much more (or less) the two parts do together than
    the sum of what each does alone. With exactly two players this equals the k-SII pair value.
    """
    return (
        value_of(table, quantity, (i, j))
        - value_of(table, quantity, (i,))
        - value_of(table, quantity, (j,))
        + value_of(table, quantity, ())
    )


def cause_superpixel(regions: np.ndarray, cause: np.ndarray) -> int:
    """The superpixel (fine player) that contains the most pixels of the planted cause."""
    return int(np.bincount(regions[cause], minlength=int(regions.max()) + 1).argmax())


def rank_of(values: np.ndarray, player: int) -> int:
    """Rank of ``player`` when the values are sorted from high to low (1 = highest value)."""
    return int((values > values[player]).sum()) + 1


# ------------------------------------------------------------------------------------ disk cache


class StoredCoalitionFeatures(CoalitionFeatures):
    """:class:`CoalitionFeatures` of a constructed image that remembers its coalitions on disk.

    The constructed images are not in the dataset, so the cache key is a hash of the pixels and the
    player map. :meth:`store` writes new coalitions back; a rerun of notebook 03-3 then needs no
    transformer passes. (A subclass instead of attaching attributes later keeps the types clear.)

    Args:
        image: ``uint8`` RGB image.
        regions: Player id per pixel.
        backbones: ``{name: FrozenViT}``.
        tag: Readable prefix of the cache file, e.g. ``"muster_test_fein"``.
    """

    def __init__(self, image: np.ndarray, regions: np.ndarray, backbones: dict, tag: str) -> None:
        """Set up the cache and load matching coalitions from disk."""
        super().__init__(image, regions, backbones)
        digest = hashlib.sha256(image.tobytes() + regions.astype(np.int64).tobytes()).hexdigest()[:16]
        self.cache_path = ARTIFACT_DIR / "ground_truth" / f"{tag}_{digest}.pt"
        self.manifest = {
            "schema": 1,
            "digest": digest,
            "models": {
                key: (getattr(b.model, "pretrained_cfg", None) or {}).get("hf_hub_id", key)
                for key, b in backbones.items()
            },
        }
        stored = load(self.cache_path)
        if stored is not None and stored["manifest"] == self.manifest:
            self.load_state(stored)

    def store(self) -> None:
        """Write all coalitions back to disk, but only if new ones were computed."""
        if self.n_forward:
            save(self.cache_path, {"manifest": self.manifest, **self.state()})
