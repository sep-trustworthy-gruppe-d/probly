"""Players for image games: grid cells or SLIC superpixels, and how they map to ViT patch tokens.

A *player* is a region of the ``224 x 224`` crop. A *coalition* is the set of visible regions.
Following Tom's comparison (branch ``tom-probly``) we offer two kinds of players:

* ``grid``: a fixed ``rows x cols`` grid (Tom: 4 x 4),
* ``slic``: SLIC superpixels that follow colour edges (Tom: 16 segments, compactness 10).

A vision transformer does not see pixels but patch tokens (``16 x 16`` pixels for ViT-S/16,
``14 x 14`` for DINOv2). To remove a player *in token space* (no fill colour), every token is
assigned to the player that covers most of its pixels. Removing a player then means hiding
all of its tokens.
"""

from __future__ import annotations

import numpy as np

CROP_SIZE = 224


def grid_regions(rows: int = 4, cols: int = 4, size: int = CROP_SIZE) -> np.ndarray:
    """Return the player id of every pixel for a ``rows x cols`` grid, numbered row by row.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.
        size: Height and width of the image in pixels.

    Returns:
        ``int64`` array of shape ``(size, size)`` with values ``0 .. rows * cols - 1``.
    """
    row_index = np.minimum(np.arange(size) * rows // size, rows - 1)
    col_index = np.minimum(np.arange(size) * cols // size, cols - 1)
    return (row_index[:, None] * cols + col_index[None, :]).astype(np.int64)


def slic_regions(image: np.ndarray, n_segments: int = 16, compactness: float = 10.0) -> np.ndarray:
    """Return exactly ``n_segments`` SLIC superpixels with the settings of Tom's comparison.

    On our ``224 x 224`` crops, SLIC with ``n_segments=16`` merges small segments and usually
    returns only 10 to 14 superpixels. So that every image has the same number of players (same
    game size, same meaning of an approximation budget), we

    1. raise the requested number until SLIC returns at least ``n_segments`` superpixels,
    2. merge the smallest superpixel into the neighbour it shares the longest border with,
       until exactly ``n_segments`` remain, and
    3. in the rare case that SLIC never reaches ``n_segments`` (very uniform images), split the largest
       superpixel in half along its longer side until there are enough.

    Args:
        image: ``uint8`` RGB image of shape ``(H, W, 3)``.
        n_segments: Number of superpixels (players) to return.
        compactness: Balance of colour similarity and spatial proximity (higher = more square).

    Returns:
        ``int64`` array of shape ``(H, W)`` with values ``0 .. n_segments - 1``.
    """
    from skimage.segmentation import slic  # noqa: PLC0415

    requested = n_segments
    for _ in range(50):
        labels = slic(image, n_segments=requested, compactness=compactness, start_label=0, channel_axis=-1)
        if len(np.unique(labels)) >= n_segments:
            break
        requested += 2
    labels = _dense(labels)
    while labels.max() + 1 > n_segments:
        labels = _merge_smallest(labels)
    while labels.max() + 1 < n_segments:
        labels = _split_largest(labels)
    return labels


def _dense(labels: np.ndarray) -> np.ndarray:
    # Renumber labels to 0 .. n - 1 without gaps.
    _, dense = np.unique(labels, return_inverse=True)
    return dense.reshape(labels.shape).astype(np.int64)


def _split_largest(labels: np.ndarray) -> np.ndarray:
    # Split the largest region in two halves along its longer extent (rows or columns).
    largest = int(np.bincount(labels.ravel()).argmax())
    rows, cols = np.nonzero(labels == largest)
    second = rows > np.median(rows) if np.ptp(rows) >= np.ptp(cols) else cols > np.median(cols)
    split = labels.copy()
    split[rows[second], cols[second]] = labels.max() + 1
    return split


def _merge_smallest(labels: np.ndarray) -> np.ndarray:
    # Merge the smallest region into the neighbour with the longest shared border.
    sizes = np.bincount(labels.ravel())
    smallest = int(sizes.argmin())
    border = np.zeros(len(sizes), dtype=np.int64)
    # Horizontal and vertical neighbour pairs of pixels.
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        touching = (a == smallest) & (b != smallest)
        np.add.at(border, b[touching], 1)
        touching = (b == smallest) & (a != smallest)
        np.add.at(border, a[touching], 1)
    border[smallest] = -1
    merged = labels.copy()
    merged[merged == smallest] = int(border.argmax())
    return _dense(merged)


def make_regions(image: np.ndarray, players: str = "slic", n_players: int = 16) -> np.ndarray:
    """Build the player map of one image.

    Args:
        image: ``uint8`` RGB crop.
        players: ``"slic"`` or ``"grid"``.
        n_players: Number of superpixels, or number of grid cells (must be a square number).

    Returns:
        Player id per pixel.
    """
    if players == "slic":
        return slic_regions(image, n_segments=n_players)
    if players == "grid":
        side = round(np.sqrt(n_players))
        if side * side != n_players:
            msg = "A grid needs a square number of players, e.g. 9 or 16."
            raise ValueError(msg)
        return grid_regions(side, side, image.shape[0])
    msg = f"Unknown players {players!r}; use 'slic' or 'grid'."
    raise ValueError(msg)


def regions_to_tokens(regions: np.ndarray, patch_size: int) -> np.ndarray:
    """Assign every patch token to the player that covers most of its pixels.

    Args:
        regions: Player id per pixel, shape ``(H, W)``; ``H`` and ``W`` must be multiples of ``patch_size``.
        patch_size: Side length of one patch token in pixels.

    Returns:
        Player id per patch token in row-major order, shape ``(H // patch_size * W // patch_size,)``.
    """
    height, width = regions.shape
    if height % patch_size or width % patch_size:
        msg = f"Image size {regions.shape} is not a multiple of the patch size {patch_size}."
        raise ValueError(msg)
    n_players = int(regions.max()) + 1
    # Cut the map into patches: (rows, patch, cols, patch) -> (rows, cols, patch * patch)
    patches = regions.reshape(height // patch_size, patch_size, width // patch_size, patch_size)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size * patch_size)
    # Count, for every patch, how many of its pixels belong to each player and take the majority.
    counts = np.stack([(patches == player).sum(axis=1) for player in range(n_players)], axis=1)
    return counts.argmax(axis=1).astype(np.int64)


def token_coalition_mask(coalitions: np.ndarray, token_players: np.ndarray) -> np.ndarray:
    """Turn player coalitions into visible/hidden flags for every patch token.

    Args:
        coalitions: Boolean array ``(n_coalitions, n_players)``; ``True`` = player is visible.
        token_players: Player id of every patch token, from :func:`regions_to_tokens`.

    Returns:
        Boolean array ``(n_coalitions, n_tokens)``; ``True`` = token is visible.
    """
    coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
    return coalitions[:, token_players]


def pixel_coalition_mask(coalitions: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Turn player coalitions into visible/hidden flags for every pixel (for image-space removal).

    Args:
        coalitions: Boolean array ``(n_coalitions, n_players)``.
        regions: Player id per pixel ``(H, W)``.

    Returns:
        Boolean array ``(n_coalitions, H, W)``.
    """
    coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
    return coalitions[:, regions]
