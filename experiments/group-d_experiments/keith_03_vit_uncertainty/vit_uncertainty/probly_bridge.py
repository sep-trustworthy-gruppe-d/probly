"""Model to probly: small heads on frozen transformer features, wrapped as probly methods.

The pipeline of this package is

    image  ->  frozen backbones (CLS features)  ->  probly predictor  ->  representer  ->  quantify

Only the small heads (``Dropout -> Linear``) are trained. The backbones stay frozen.

Every probly predictor in this package reads the **same input**: the CLS features of all
backbones concatenated (``vit`` then ``dinov2``, 384 + 384 = 768 numbers). A member that belongs
to one backbone cuts out its own block with :class:`FeatureSlice`. This makes members on
*different* backbones possible inside one probly ensemble, which ``probly.method.ensemble``
does not support directly (it copies one base model). We learned in the DermaMNIST series that
exactly these mixed ensembles show epistemic uncertainty; heads on one frozen network hardly do.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    import numpy as np

# Training settings of Dan's VGG16 heads, so that differences come from the backbone, not from tuning.
HEAD_CONFIG = {
    "dropout": 0.5,
    "batch_size": 64,
    "max_epochs": 30,
    "patience": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}


@dataclass(frozen=True)
class FeatureLayout:
    """Where each backbone's features sit inside the concatenated feature vector.

    Attributes:
        names: Backbone names in concatenation order.
        dims: Feature dimension of each backbone, same order.
    """

    names: tuple[str, ...]
    dims: tuple[int, ...]

    @classmethod
    def from_backbones(cls, backbones: dict) -> FeatureLayout:
        """Build the layout from a ``{name: FrozenViT}`` dictionary (its order is the concatenation order)."""
        return cls(tuple(backbones), tuple(int(b.feature_dim) for b in backbones.values()))

    @property
    def total(self) -> int:
        """Length of the concatenated vector."""
        return int(sum(self.dims))

    def span(self, name: str) -> tuple[int, int]:
        """``[start, end)`` columns of one backbone."""
        start = 0
        for current, dim in zip(self.names, self.dims, strict=True):
            if current == name:
                return start, start + dim
            start += dim
        msg = f"Backbone {name!r} is not in the layout {self.names}."
        raise KeyError(msg)

    def concat(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate per-backbone features in layout order."""
        return torch.cat([features[name] for name in self.names], dim=-1)


class FeatureSlice(nn.Module):
    """Keep only the columns of one backbone from the concatenated feature vector."""

    def __init__(self, name: str, start: int, end: int) -> None:
        """Remember which columns belong to the backbone ``name``."""
        super().__init__()
        self.name, self.start, self.end = name, start, end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x[:, start:end]``."""
        return x[:, self.start : self.end]

    def extra_repr(self) -> str:
        """Shown when the module is printed."""
        return f"{self.name}, columns {self.start}:{self.end}"


def seed_all(seed: int) -> None:
    """Seed Python and PyTorch (NumPy code in this package uses explicit generators)."""
    random.seed(seed)
    torch.manual_seed(seed)


def make_head(feature_dim: int, n_classes: int, *, dropout: float = 0.5, seed: int = 0) -> nn.Sequential:
    """Build ``Dropout -> Linear``; ``seed`` fixes the initial weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, n_classes))


def train_head(
    head: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    *,
    config: dict | None = None,
    seed: int = 0,
) -> dict:
    """Train one head with AdamW and keep the epoch with the lowest validation cross-entropy.

    Args:
        head: Initialized head (trained in place).
        train: Training features and labels.
        val: Validation features and labels.
        config: Settings; defaults to :data:`HEAD_CONFIG` (Dan's settings).
        seed: Seed for dropout masks and minibatch order.

    Returns:
        ``state_dict`` of the best epoch, ``best_epoch``, ``best_val_loss``, and the ``history``.
    """
    config = HEAD_CONFIG | (config or {})
    seed_all(seed)
    loader = DataLoader(
        TensorDataset(*train),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best = {"state_dict": None, "best_epoch": 0, "best_val_loss": float("inf"), "history": []}
    patience = 0
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(head(features), labels).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(val[0])
            val_loss = criterion(logits, val[1]).item()
            val_acc = (logits.argmax(1) == val[1]).float().mean().item()
        best["history"].append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        if val_loss < best["best_val_loss"]:
            best.update(best_val_loss=val_loss, best_epoch=epoch)
            best["state_dict"] = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config["patience"]:
                break
    head.load_state_dict(best["state_dict"])
    head.eval()
    return best


def member(layout: FeatureLayout, backbone: str, state_dict: dict, *, dropout: float = 0.5) -> nn.Sequential:
    """One trained ensemble member ``FeatureSlice -> Dropout -> Linear`` reading the concatenated input."""
    start, end = layout.span(backbone)
    n_classes = state_dict["1.weight"].shape[0]
    head = make_head(end - start, n_classes, dropout=dropout)
    head.load_state_dict(state_dict)
    return nn.Sequential(FeatureSlice(backbone, start, end), *head).eval()


def build_probly_ensemble(members: list[nn.Module]) -> nn.ModuleList:
    """Wrap trained, possibly different members as one probly ensemble.

    ``probly.method.ensemble`` builds ``num_members`` copies of one base model. We let probly build
    that container (so that probly registers it as an ensemble predictor) and then put our trained
    members into its slots, each registered as a logit classifier.
    """
    from probly.method import cast, ensemble  # noqa: PLC0415

    container = ensemble(members[0], num_members=len(members), reset_params=False, predictor_type="logit_classifier")
    for index, module in enumerate(members):
        container[index] = cast(module, predictor_type="logit_classifier")
    return container.eval()


def build_probly_dropout(member_module: nn.Module, p: float = 0.5) -> nn.Module:
    """MC Dropout with probly: the head's own dropout layer stays active at prediction time."""
    from probly.method import dropout  # noqa: PLC0415

    return dropout(member_module, p=p, predictor_type="logit_classifier").eval()


def build_methods(
    layout: FeatureLayout,
    states: dict[str, list[dict]],
    *,
    n_members: int = 6,
    dropout: float = 0.5,
) -> dict[str, nn.Module]:
    """Build the probly methods compared in this package.

    * ``"Ensemble ViT"`` (variant a): ``n_members`` heads on the ViT features,
    * ``"MC Dropout ViT"`` (variant a): the first ViT head with active dropout,
    * ``"Gemischt"`` (variant b): half of the heads on ViT, half on DINOv2.

    All ensembles have the same number of members, because EU also depends on that number.

    Args:
        layout: Feature layout of the backbones.
        states: Trained head weights per backbone, indexed by seed.
        n_members: Members per ensemble (even).
        dropout: Dropout probability of the heads.

    Returns:
        Name -> probly predictor, all in float64 (keeps TU = AU + EU exact for tiny EU values).
    """
    half = n_members // 2

    def heads(name: str, count: int) -> list[nn.Module]:
        return [member(layout, name, states[name][seed], dropout=dropout) for seed in range(count)]

    methods = {
        "Ensemble ViT": build_probly_ensemble(heads("vit", n_members)),
        "MC Dropout ViT": build_probly_dropout(member(layout, "vit", states["vit"][0], dropout=dropout), p=dropout),
        "Gemischt": build_probly_ensemble(heads("vit", half) + heads("dinov2", half)),
    }
    return {name: predictor.double() for name, predictor in methods.items()}


def is_sampling_method(predictor: nn.Module) -> bool:
    """``True`` for MC Dropout predictors (they need ``num_samples`` in the representer)."""
    return not isinstance(predictor, nn.ModuleList)


def representation(predictor: nn.Module, features: torch.Tensor, *, num_samples: int = 50, seed: int = 0) -> Any:  # noqa: ANN401
    """Probly representation of fixed features (seeded, so MC Dropout is reproducible)."""
    from probly.representer import representer  # noqa: PLC0415

    rep = representer(predictor, num_samples=num_samples) if is_sampling_method(predictor) else representer(predictor)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        return rep.represent(features.double())


def uncertainty(
    predictor: nn.Module, features: torch.Tensor, *, batch_size: int = 1024, **kwargs: int
) -> dict[str, np.ndarray]:
    """Mean probabilities and probly's TU/AU/EU (nats) for fixed features.

    Ensembles are evaluated in batches. MC Dropout is evaluated **row by row with the same seed**:
    probly draws one dropout mask per row and sample, so in a batch the masks of a row depend on its
    position. Row by row, every input (every image, every coalition) sees the same 50 masks. Then a
    game value v(S) is a fixed number and not a random draw, which Shapley values need.
    """
    from probly.quantification import quantify  # noqa: PLC0415

    if is_sampling_method(predictor):
        batch_size = 1
    out = {"probabilities": [], "TU": [], "AU": [], "EU": []}
    for batch in features.split(batch_size):
        rep = representation(predictor, batch, **kwargs)
        decomposition = quantify(rep)
        out["probabilities"].append(torch.as_tensor(rep.sample_mean().probabilities))
        out["TU"].append(torch.as_tensor(decomposition.total))
        out["AU"].append(torch.as_tensor(decomposition.aleatoric))
        out["EU"].append(torch.as_tensor(decomposition.epistemic))
    return {key: torch.cat(value).double().numpy() for key, value in out.items()}
