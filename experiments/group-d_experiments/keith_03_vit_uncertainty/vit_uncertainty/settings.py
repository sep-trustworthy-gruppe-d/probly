"""All fixed settings of the experiments in one place (so the notebooks and scripts agree)."""

from __future__ import annotations

SEED = 0

# Players (as in Tom's comparison): 16 SLIC superpixels, compactness 10; grid 4 x 4 for comparison.
N_PLAYERS = 16

# Explanations: k-SII up to order 2 with a budget of 1000 coalitions (as Tom).
INDEX = "k-SII"
MAX_ORDER = 2
BUDGET = 1000

# probly methods: 6 members per ensemble (3 + 3 in the mixed one), 50 MC Dropout samples.
N_MEMBERS = 6
MC_SAMPLES = 50

# Masked training data ("surrogate" idea of ViT Shapley): random SLIC coalitions per image.
MASKED_PER_IMAGE_TRAIN = 8
MASKED_PER_IMAGE_VAL = 4

# Subset of the 37-image explanation cohort for the expensive comparisons: every third breed.
SUBSET_POSITIONS = list(range(0, 37, 3))

# Learned explainer vs. KernelSHAP: budgets of the comparison curve and of the reference.
SV_BUDGETS = [64, 128, 256, 512]
SV_REFERENCE_BUDGET = 2048
