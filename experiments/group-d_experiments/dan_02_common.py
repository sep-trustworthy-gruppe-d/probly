"""Shared loading, image geometry, and small-head training for Dan's Pet notebooks.

The experiments and their probly/shapiq calls live in the notebooks. This module
only keeps their mechanical setup consistent.
"""

from __future__ import annotations

import hashlib
import json
import random
from importlib.metadata import version
from itertools import batched
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import InterpolationMode, functional as TF

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "oxford-pets-vgg16"
BASELINE_NOTEBOOK = "dan_02-0_oxford-pets-vgg16-baseline.ipynb"
WEIGHTS = VGG16_Weights.IMAGENET1K_V1
PREPROCESS = WEIGHTS.transforms()


def versions() -> dict[str, str]:
    """Return installed versions relevant to reproducible experiment artifacts."""
    return {
        name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "probly", "shapiq", "scikit-learn")
    }


def select_device() -> torch.device:
    """Select CUDA, MPS, or CPU for the frozen image encoder."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch without changing experiment settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_torch(path: Path | str, payload: dict) -> None:
    """Atomically save tensors and plain Python metadata to a local artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_torch(path: Path | str, prerequisite: str = BASELINE_NOTEBOOK) -> dict:
    """Load a local tensor artifact, naming its prerequisite when it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run {prerequisite} first.")
    return torch.load(path, map_location="cpu", weights_only=True)


def state_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, types, shapes, and contents rather than checkpoint dates."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_head(
    config: dict,
    n_classes: int = 37,
    *,
    state_dict: dict | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    """Construct the baseline Dropout/Linear head, optionally restoring weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(config["feature_dim"], n_classes),
        )
    if state_dict is not None:
        head.load_state_dict(state_dict)
    return head.to(device=device, dtype=dtype).eval()


def train_head(
    head: nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    config: dict,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train one initialized head and restore its minimum-validation-loss state.

    Args:
        head: Initialized head; its parameters are not reinitialized here.
        train_features: Frozen training features in sample order.
        train_labels: Training breed indices.
        val_features: Frozen validation features.
        val_labels: Validation breed indices.
        config: Baseline optimizer, batching, and early-stopping configuration.
        seed: Seed for training dropout and shuffled minibatches.
        device: Device for the small head.
        verbose: Whether to print one progress line per epoch.

    Returns:
        The best CPU state, training history, selected epoch, loss, and seed.
    """
    seed_all(seed)
    head.to(device)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    history, best_state = [], None
    best_loss, best_epoch, patience = float("inf"), 0, 0
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        train_loss, train_correct = 0.0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.detach().argmax(1) == labels).sum().item()
        head.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = head(features)
                val_loss += criterion(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
        record = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_labels),
            "val_loss": val_loss / len(val_labels),
            "train_accuracy": train_correct / len(train_labels),
            "val_accuracy": val_correct / len(val_labels),
        }
        if not all(np.isfinite(value) for value in record.values()):
            raise ValueError("Training produced a non-finite metric.")
        history.append(record)
        if verbose:
            print(
                f"Epoche {epoch:02d} | Loss {record['train_loss']:.4f}/{record['val_loss']:.4f}"
                f" | Accuracy {record['train_accuracy']:.1%}/{record['val_accuracy']:.1%} (Train/Val)"
            )
        if record["val_loss"] < best_loss:
            best_loss, best_epoch, patience = record["val_loss"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        else:
            patience += 1
            if patience >= config["patience"]:
                if verbose:
                    print(f"Early Stopping nach {epoch} Epochen.")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid head.")
    head.load_state_dict(best_state)
    head.eval()
    return {
        "state_dict": best_state,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "seed": seed,
    }


def head_logits(head: nn.Module, features: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Predict logits in bounded batches and return ordinary CPU tensors."""
    parameter = next(head.parameters())
    head.eval()
    outputs = []
    with torch.no_grad():
        for batch in features.split(batch_size):
            outputs.append(head(batch.to(device=parameter.device, dtype=parameter.dtype)).cpu())
    return torch.cat(outputs)


def load_baseline() -> dict:
    """Load and cross-check the baseline checkpoint, feature caches, and predictions."""
    checkpoint = load_torch(ARTIFACT_DIR / "best_head.pt")
    trainval = load_torch(ARTIFACT_DIR / "trainval_vgg16_features.pt")
    test = load_torch(ARTIFACT_DIR / "test_vgg16_features.pt")
    predictions = load_torch(ARTIFACT_DIR / "baseline_predictions.pt")
    fingerprint = state_fingerprint(checkpoint["head_state_dict"])
    if predictions["baseline_fingerprint"] != fingerprint:
        raise ValueError(f"Baseline predictions are stale. Run {BASELINE_NOTEBOOK} again.")
    for cache in (trainval, test):
        if cache["manifest"]["classes"] != checkpoint["classes"]:
            raise ValueError("Cached class order differs from the baseline checkpoint.")
        if cache["manifest"]["preprocessing"] != checkpoint["preprocessing"]:
            raise ValueError("Cached preprocessing differs from the baseline checkpoint.")
        if not torch.isfinite(cache["features"]).all():
            raise ValueError("Non-finite baseline feature cache.")
    if predictions["image_ids"] != test["manifest"]["image_ids"]:
        raise ValueError("Baseline prediction IDs do not match the test cache.")
    if not torch.equal(predictions["labels"], test["labels"]):
        raise ValueError("Baseline prediction labels do not match the test cache.")
    positions = {name: index for index, name in enumerate(trainval["manifest"]["image_ids"])}
    train_indices = torch.tensor([positions[name] for name in checkpoint["train_image_ids"]])
    val_indices = torch.tensor([positions[name] for name in checkpoint["validation_image_ids"]])
    if set(train_indices.tolist()) & set(val_indices.tolist()):
        raise ValueError("Training and validation indices overlap.")
    return {
        "checkpoint": checkpoint,
        "trainval": trainval,
        "test": test,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "baseline_predictions": predictions,
    }


def load_backbone(device: torch.device | str | None = None) -> nn.Module:
    """Restore the frozen VGG encoder from the baseline's local weight cache."""
    torch.hub.set_dir(str(ARTIFACT_DIR / "torch_hub"))
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS.url.rsplit("/", 1)[-1]
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing VGG weights. Run {BASELINE_NOTEBOOK} first.")
    model = vgg16(weights=WEIGHTS)
    model.classifier[-1] = nn.Identity()
    return model.requires_grad_(False).eval().to(select_device() if device is None else device)


def load_crop(image_id: str) -> np.ndarray:
    """Load the RGB crop seen by VGG, without normalizing or cropping it twice."""
    path = DATA_ROOT / "oxford-iiit-pet" / "images" / f"{image_id}.jpg"
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = TF.resize(
            image, PREPROCESS.resize_size, interpolation=PREPROCESS.interpolation, antialias=PREPROCESS.antialias
        )
        return np.asarray(TF.center_crop(image, PREPROCESS.crop_size)).copy()


def load_trimap(image_id: str) -> np.ndarray:
    """Align raw trimap labels with the model crop using nearest-neighbor geometry."""
    path = DATA_ROOT / "oxford-iiit-pet" / "annotations" / "trimaps" / f"{image_id}.png"
    with Image.open(path) as trimap:
        trimap = TF.resize(trimap, PREPROCESS.resize_size, interpolation=InterpolationMode.NEAREST)
        values = np.asarray(TF.center_crop(trimap, PREPROCESS.crop_size)).astype(np.int64)
    if values.shape != (224, 224) or not np.isin(values, [1, 2, 3]).all():
        raise ValueError(f"Invalid aligned trimap for {image_id}.")
    return values


def rgb_tensor(crop: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert an RGB uint8 HWC crop or float CHW crop to a CPU [0, 1] tensor."""
    if isinstance(crop, torch.Tensor):
        tensor = crop.detach().cpu().float()
    else:
        array = np.asarray(crop)
        if array.dtype != np.uint8 or array.shape != (224, 224, 3):
            raise ValueError("RGB arrays must be uint8 with shape (224, 224, 3).")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255
    if tensor.shape != (3, 224, 224) or tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("RGB tensors must have shape (3, 224, 224) and values in [0, 1].")
    return tensor


def extract_crops(backbone: nn.Module, crops, batch_size: int = 4) -> torch.Tensor:
    """Encode an iterable of already-cropped RGB images once in bounded batches."""
    parameter = next(backbone.parameters())
    backbone.eval()
    outputs = []
    with torch.no_grad():
        for chunk in batched(crops, batch_size):
            images = torch.stack([rgb_tensor(image) for image in chunk])
            images = TF.normalize(images, PREPROCESS.mean, PREPROCESS.std)
            features = backbone(images.to(device=parameter.device, dtype=parameter.dtype)).cpu().float()
            if not torch.isfinite(features).all():
                raise ValueError("VGG produced non-finite features.")
            outputs.append(features)
    return torch.cat(outputs)


def write_selection(test_cache: dict, seed: int = 0) -> dict:
    """Persist label-only random cohorts shared by corruption and explanation studies."""
    rng = np.random.default_rng(seed)
    labels = test_cache["labels"].numpy()
    image_ids = test_cache["manifest"]["image_ids"]
    corruption, explanation = [], []
    for breed in range(len(test_cache["manifest"]["classes"])):
        selected = rng.choice(np.flatnonzero(labels == breed), size=10, replace=False).tolist()
        corruption.extend(selected)
        explanation.append(selected[0])
    selection = {
        "schema_version": 1,
        "seed": seed,
        "corruption_indices": corruption,
        "explanation_indices": explanation,
        "corruption_ids": [image_ids[index] for index in corruption],
        "explanation_ids": [image_ids[index] for index in explanation],
    }
    path = ARTIFACT_DIR / "series_selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Existing experiment cohorts differ from the current labels or seed.")
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return selection


def load_selection() -> dict:
    """Load the cohorts exported by the baseline without making a new selection."""
    path = ARTIFACT_DIR / "series_selection.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared cohorts. Run {BASELINE_NOTEBOOK} first.")
    return json.loads(path.read_text())


def diagnostic_cases(bundle: dict, selection: dict) -> list[dict]:
    """Select explicit illustrative cases inside the fixed 37-image cohort."""
    probabilities = bundle["baseline_predictions"]["probabilities"]
    labels = bundle["test"]["labels"]
    image_ids = bundle["test"]["manifest"]["image_ids"]
    cohort = torch.tensor(selection["explanation_indices"])
    confidence, prediction = probabilities.max(1)
    correct = prediction[cohort] == labels[cohort]
    rules = [
        ("Sicher richtig", cohort[correct], True),
        ("Unsicher richtig", cohort[correct], False),
        ("Sicher falsch", cohort[~correct], True),
    ]
    cases = []
    for role, candidates, descending in rules:
        if not len(candidates):
            continue
        order = torch.argsort(confidence[candidates], descending=descending, stable=True)
        index = int(candidates[order[0]])
        target = int(prediction[index])
        comparison = (
            int(labels[index]) if target != int(labels[index]) else int(probabilities[index].topk(2).indices[1])
        )
        cases.append(
            {
                "role": role,
                "index": index,
                "image_id": image_ids[index],
                "target_class": target,
                "true_class": int(labels[index]),
                "comparison_class": comparison,
                "confidence": float(confidence[index]),
            }
        )
    return cases
