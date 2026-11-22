"""Bridge: PyTorch VGG-16 transformed with ``probly`` -> ``vision_uncertainty``.

This file is the *only* place that touches torch / torchvision / probly; the explanation
module stays dependency-light. It turns a stochastic model (MC-dropout) or an ensemble into
the ``predict_fn(images (B,H,W,C)) -> probs (B,M,K)`` contract expected by
:func:`vision_uncertainty.explain_uncertainty`.

NOTE: written against probly's ``transformation`` API (``dropout`` / ``ensemble``); check the
function names / signatures against your installed probly version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vision_uncertainty import explain_uncertainty

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_probly_vgg16(kind: str = "dropout", p: float = 0.2, num_members: int = 5) -> nn.Module | nn.ModuleList:
    """Build an uncertainty-aware VGG-16 model using Probly transformations."""
    from torchvision.models import VGG16_Weights, vgg16  # noqa: PLC0415

    base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
    if kind == "dropout":
        from probly.transformation import dropout  # noqa: PLC0415

        # Replace VGG's built-in dropout so the requested probability is the only one used.
        base.classifier = nn.Sequential(
            *(nn.Identity() if isinstance(layer, nn.Dropout) else layer for layer in base.classifier)
        )
        base.classifier = dropout(base.classifier, p=p)
        return base
    if kind == "ensemble":
        from probly.transformation import ensemble  # noqa: PLC0415

        return ensemble(base, num_members=num_members)
    msg = f"Unknown kind '{kind}'. Use 'dropout' or 'ensemble'."
    raise ValueError(msg)


def make_predict_fn(
    model: nn.Module | nn.ModuleList,
    n_samples: int = 10,
    device: str | torch.device | None = None,
    class_subset: list[int] | None = None,
):
    """Wrap a probly model as ``predict_fn((B,H,W,C) in [0,1]) -> (B, M, K)`` probabilities.

    * ``nn.ModuleList`` / list  -> one forward pass per ensemble member (``M = len(model)``).
    * single ``nn.Module``      -> ``n_samples`` stochastic passes with only Dropout layers in
      train mode (MC-dropout), everything else (e.g. BatchNorm) stays in eval mode. VGG feature
      extraction is cached once per input batch and only its classifier is repeated.
    * ``class_subset``          -> restrict to K classes and renormalise (keeps entropies and
      credal sets meaningful when only a few classes, e.g. dog vs. cat, matter).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    members = list(model) if isinstance(model, (nn.ModuleList, list, tuple)) else [model]
    for m in members:
        m.to(device).eval()

    @torch.no_grad()
    def predict_fn(images: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        x = (x - mean) / std
        outs = []
        if len(members) > 1:
            outs = [torch.softmax(m(x), dim=-1) for m in members]
        else:
            net = members[0]
            is_vgg = all(hasattr(net, name) for name in ("features", "avgpool", "classifier"))
            if is_vgg:
                features = net.avgpool(net.features(x))
                features = torch.flatten(features, 1)
                classifier = net.classifier
            for mod in net.modules():
                if isinstance(mod, nn.modules.dropout._DropoutNd):
                    mod.train()
            if is_vgg:
                logits = [classifier(features) for _ in range(n_samples)]
            else:
                logits = [net(x) for _ in range(n_samples)]
            outs = [torch.softmax(output, dim=-1) for output in logits]
            net.eval()
        probs = torch.stack(outs, dim=1)  # (B, M, 1000)
        if class_subset is not None:
            probs = probs[..., class_subset]
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs.cpu().numpy().astype(float)

    return predict_fn


def load_image(path: str, size: int = 224) -> np.ndarray:
    """Load an RGB image as ``(size, size, 3)`` float array in [0, 1]."""
    from PIL import Image  # noqa: PLC0415

    return np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=float) / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain predictive uncertainty for an image with VGG-16.")
    parser.add_argument(
        "image",
        nargs="?",
        default=Path(__file__).resolve().parent / "Pictures" / "DogCat.jpeg",
        type=Path,
        help="input image (default: this folder's Pictures/DogCat.jpeg)",
    )
    parser.add_argument("--kind", choices=("dropout", "ensemble"), default="dropout")
    parser.add_argument("--samples", type=int, default=8, help="MC-dropout samples")
    parser.add_argument("--budget", type=int, default=600, help="maximum coalition evaluations")
    parser.add_argument("--device", help="torch device, for example cpu or cuda")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")

    image = load_image(str(args.image))
    predict = make_predict_fn(
        build_probly_vgg16(args.kind, p=0.2),
        n_samples=args.samples,
        device=args.device,
        class_subset=[207, 281],  # golden retriever, tabby cat (example ImageNet ids)
    )
    exp = explain_uncertainty(
        image,
        predict,
        grid_shape=(4, 4),
        measure="epistemic",
        baseline="blur",
        budget=args.budget,
        batch_size=16,
    )
    print("total epistemic uncertainty:", exp.full_value)
    for i, j, v in exp.top_interactions(5):
        print(f"patches {exp.grid.to_rc(i)} x {exp.grid.to_rc(j)}: phi_ij = {v:+.4f}")
