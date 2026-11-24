"""vit_uncertainty: explain prediction *and* uncertainty of vision transformers with probly and shapiq.

Prototype of a shapiq extension (Keith, group D, "Trustworthy Machine Learning with Python").

The pipeline in four steps::

    from vit_uncertainty import (
        OxfordPets, load_backbones, build_methods, UncertaintyExplainer, plot_prediction_and_uncertainty,
    )

    backbones = load_backbones()                       # 1. model: frozen ViT-S/16 and DINOv2-S/14
    methods = build_methods(layout, trained_heads)     # 2. model -> probly (ensembles, MC Dropout)
    explainer = UncertaintyExplainer(backbones, methods["Gemischt"])   # 3. a shapiq.Explainer
    explanation = explainer.explain(image)             #    P, TU, AU, EU as shapiq InteractionValues
    plot_prediction_and_uncertainty(explanation)       # 4. explanation of both, side by side

Optional, step 5: :class:`AmortizedUncertaintyExplainer` learns to output the Shapley values of
P, TU, AU, and EU in one forward pass (ViT Shapley for uncertainty).

Modules:

* :mod:`~vit_uncertainty.data`: Oxford-IIIT Pet with Dan's splits and cohorts.
* :mod:`~vit_uncertainty.backbones`: frozen transformers that can hide patch tokens.
* :mod:`~vit_uncertainty.players`: grid and SLIC players and their patch tokens (as in Tom's comparison).
* :mod:`~vit_uncertainty.probly_bridge`: heads and probly methods (also mixed ensembles).
* :mod:`~vit_uncertainty.removal`: evaluating coalitions once, with a cache.
* :mod:`~vit_uncertainty.game`: :class:`UncertaintyGame`, a ``shapiq.Game``.
* :mod:`~vit_uncertainty.explainer`: :class:`UncertaintyExplainer`, a ``shapiq.Explainer``.
* :mod:`~vit_uncertainty.plot`: prediction and uncertainty side by side.
* :mod:`~vit_uncertainty.amortized`: the learned explainer (ViT Shapley for uncertainty).
* :mod:`~vit_uncertainty.features`: feature caches for the notebooks.
"""

from .amortized import AmortizedUncertaintyExplainer
from .backbones import BACKBONES, FrozenViT, load_backbones
from .data import OxfordPets, default_data_root
from .explainer import UncertaintyExplainer, UncertaintyExplanation
from .game import QUANTITIES, UncertaintyGame
from .players import grid_regions, make_regions, regions_to_tokens, slic_regions
from .plot import plot_comparison, plot_pairs, plot_prediction_and_uncertainty
from .probly_bridge import FeatureLayout, build_methods, build_probly_dropout, build_probly_ensemble, uncertainty
from .removal import CoalitionFeatures

__all__ = [
    "BACKBONES",
    "QUANTITIES",
    "AmortizedUncertaintyExplainer",
    "CoalitionFeatures",
    "FeatureLayout",
    "FrozenViT",
    "OxfordPets",
    "UncertaintyExplainer",
    "UncertaintyExplanation",
    "UncertaintyGame",
    "build_methods",
    "build_probly_dropout",
    "build_probly_ensemble",
    "default_data_root",
    "grid_regions",
    "load_backbones",
    "make_regions",
    "plot_comparison",
    "plot_pairs",
    "plot_prediction_and_uncertainty",
    "regions_to_tokens",
    "slic_regions",
    "uncertainty",
]
