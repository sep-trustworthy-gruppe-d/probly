"""Oxford-IIIT Pet with exactly the splits and cohorts of Dan's VGG16 series.

Dan's series (branch ``dan/oxford-pets-vgg16``) defines:

* the official ``trainval`` (3680 images) and ``test`` (3669 images) lists,
* a stratified 80/20 split of ``trainval`` into training and validation (seed 0),
* the model input: resize the short side to 256 pixels, then a 224 x 224 center crop,
* an explanation cohort of one test image per breed (37 images) and a corruption
  cohort of ten test images per breed, both drawn only from labels with seed 0.

We rebuild all of this here without importing Dan's code, so that results stay directly
comparable with his VGG16 notebooks while this package remains independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import InterpolationMode, functional as TF

# The crop that every model in this package sees (identical to Dan's VGG16 preprocessing).
RESIZE_SIZE = 256
CROP_SIZE = 224
SEED = 0


def default_data_root() -> Path:
    """Return ``<repo>/data``, the folder Dan's notebooks use for Oxford-IIIT Pet."""
    # This file lives in <repo>/experiments/group-d_experiments/keith_03_vit_uncertainty/vit_uncertainty/.
    return Path(__file__).resolve().parents[4] / "data"


@dataclass(frozen=True)
class OxfordPets:
    """Index of the Oxford-IIIT Pet files with Dan's splits.

    The dataset itself is downloaded by ``torchvision.datasets.OxfordIIITPet(root, download=True)``
    into ``root / "oxford-iiit-pet"``; this class only reads the files.

    Attributes:
        root: Folder that contains ``oxford-iiit-pet/``.
    """

    root: Path

    @property
    def folder(self) -> Path:
        """Folder with ``images/`` and ``annotations/``."""
        return Path(self.root) / "oxford-iiit-pet"

    def _read_list(self, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        # Each line of trainval.txt / test.txt is "<image id> <class id> <species id> <breed id>",
        # with 1-based class ids. Comment lines start with "#".
        path = self.folder / "annotations" / f"{split}.txt"
        if not path.is_file():
            msg = (
                f"{path} fehlt. Bitte einmal torchvision.datasets.OxfordIIITPet('{self.root}', download=True) "
                "ausführen oder Dans Baseline-Notebook laufen lassen."
            )
            raise FileNotFoundError(msg)
        ids, labels, species = [], [], []
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            image_id, class_id, species_id, _ = line.split()
            ids.append(image_id)
            labels.append(int(class_id) - 1)
            species.append(int(species_id) - 1)
        return ids, np.asarray(labels, dtype=np.int64), np.asarray(species, dtype=np.int64)

    @cached_property
    def trainval(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Image ids, breed labels (0..36), and species labels (0 = cat, 1 = dog) of ``trainval``."""
        return self._read_list("trainval")

    @cached_property
    def test(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Image ids, breed labels, and species labels of ``test``."""
        return self._read_list("test")

    @cached_property
    def classes(self) -> list[str]:
        """Breed names in label order, written like torchvision (e.g. ``"American Bulldog"``)."""
        ids, labels, _ = self.trainval
        names = {}
        for image_id, label in zip(ids, labels, strict=True):
            # "american_bulldog_10" -> "American Bulldog"
            names[int(label)] = " ".join(part.title() for part in image_id.rsplit("_", 1)[0].split("_"))
        return [names[index] for index in range(len(names))]

    @cached_property
    def train_val_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions in ``trainval`` of Dan's training and validation images (stratified 80/20, seed 0)."""
        _, labels, _ = self.trainval
        train, val = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=SEED, stratify=labels)
        return np.asarray(train), np.asarray(val)

    @cached_property
    def cohorts(self) -> dict[str, list[int]]:
        """Dan's label-only test cohorts (positions in ``test``).

        For every breed, ten test images are drawn with ``np.random.default_rng(0)``. All 370 form
        the corruption cohort; the first image of every breed forms the 37-image explanation cohort.
        """
        _, labels, _ = self.test
        rng = np.random.default_rng(SEED)
        corruption, explanation = [], []
        for breed in range(len(self.classes)):
            selected = rng.choice(np.flatnonzero(labels == breed), size=10, replace=False).tolist()
            corruption.extend(selected)
            explanation.append(selected[0])
        return {"corruption": corruption, "explanation": explanation}

    def image_path(self, image_id: str) -> Path:
        """Path of the JPEG of one image id."""
        return self.folder / "images" / f"{image_id}.jpg"

    def load_crop(self, image_id: str) -> np.ndarray:
        """Load the ``224 x 224`` RGB crop that all models see (same pixels as in Dan's series).

        Returns:
            ``uint8`` array of shape ``(224, 224, 3)``.
        """
        with Image.open(self.image_path(image_id)) as raw:
            rgb = raw.convert("RGB")
        resized = TF.resize(rgb, RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
        return np.asarray(TF.center_crop(resized, CROP_SIZE)).copy()

    def load_trimap(self, image_id: str) -> np.ndarray:
        """Load the pet/background/border trimap aligned with :meth:`load_crop`.

        Returns:
            ``int64`` array of shape ``(224, 224)`` with 1 = pet, 2 = background, 3 = border.
        """
        path = self.folder / "annotations" / "trimaps" / f"{image_id}.png"
        with Image.open(path) as raw:
            resized = TF.resize(raw, RESIZE_SIZE, interpolation=InterpolationMode.NEAREST)
            values = np.asarray(TF.center_crop(resized, CROP_SIZE)).astype(np.int64)
        if values.shape != (CROP_SIZE, CROP_SIZE) or not np.isin(values, [1, 2, 3]).all():
            msg = f"Invalid trimap for {image_id}."
            raise ValueError(msg)
        return values

    def load_head_box(self, image_id: str, *, min_inside: float = 0.9) -> tuple[int, int, int, int] | None:
        """Load the head bounding box of the pet, moved into the coordinates of :meth:`load_crop`.

        Oxford-IIIT Pet has a head box (``annotations/xmls``) for most ``trainval`` images (3686 files),
        but none for the test images. The box is given in pixels of the original JPEG. We apply the same
        resize (short side 256) and center crop (224) as :meth:`load_crop`, so the box lies on the pixels
        the models see. The XML coordinates are 1-based; the shift of at most one pixel is ignored.

        Args:
            image_id: Image id, e.g. ``"Abyssinian_1"``.
            min_inside: Smallest share of the box area that must lie inside the crop. Boxes cut off
                by the center crop are rejected, because half a head is not a head.

        Returns:
            ``(x0, y0, x1, y1)`` in crop pixels (``x1``/``y1`` exclusive), or ``None`` if there is no box
            or too little of it lies inside the crop.
        """
        path = self.folder / "annotations" / "xmls" / f"{image_id}.xml"
        if not path.is_file():
            return None
        # The XML files are part of the official dataset download (trusted input).
        box = ET.parse(path).getroot().find("object/bndbox")  # noqa: S314
        if box is None:
            return None
        x0, y0, x1, y1 = (float(box.findtext(key) or "nan") for key in ("xmin", "ymin", "xmax", "ymax"))
        with Image.open(self.image_path(image_id)) as raw:
            width, height = raw.size
        new_width, new_height = _resized_size(width, height)
        left = round((new_width - CROP_SIZE) / 2)  # same rounding as torchvision's center_crop
        top = round((new_height - CROP_SIZE) / 2)
        sx, sy = new_width / width, new_height / height
        moved = np.array([x0 * sx - left, y0 * sy - top, x1 * sx - left, y1 * sy - top])
        clipped = np.clip(moved, 0, CROP_SIZE)
        area = (moved[2] - moved[0]) * (moved[3] - moved[1])
        inside = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        if not np.isfinite(area) or area <= 0 or inside / area < min_inside:
            return None
        bx0, by0, bx1, by1 = (round(float(value)) for value in clipped)
        return bx0, by0, bx1, by1


def _resized_size(width: int, height: int) -> tuple[int, int]:
    # Size after torchvision's resize(256) of a PIL image: the short side becomes 256 and the long
    # side int(256 * long / short), exactly as in torchvision.transforms.functional.resize.
    if width <= height:
        return RESIZE_SIZE, int(RESIZE_SIZE * height / width)
    return int(RESIZE_SIZE * width / height), RESIZE_SIZE
