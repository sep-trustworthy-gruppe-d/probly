# Uncertainty Interactions and Shapley Uncertainty Maps

Onboarding notebook for the shapiq-probly project (written in German, structured as a presentation).

It explains how Shapley interactions reveal which feature *combinations* drive predictive uncertainty, separately for
aleatoric (AU) and epistemic (EU) uncertainty, and visualizes the explanations as maps over the feature space.

Contents:

- Shapley values, Moebius transform, and interaction indices (SII, k-SII, STII, FSII) on toy games
- Entropy decomposition TU = AU + EU for an ensemble
- An uncertainty game (variant A from the project plan) on TwoMoons with a bootstrap ensemble of 10 MLPs
- Shapley uncertainty maps: main effects and interaction for AU and EU over a 2D grid
- Design question D1 (average then measure vs. measure then average) and D2 (imputation artifact)
- A 4-feature benchmark with built-in ground truth (EU interaction, noise feature, dummy feature)
- Pros, cons, and relevance for the project

All Shapley quantities are computed exactly with numpy, so neither shapiq nor probly is required yet. The appendix
sketches how the value function maps onto `shapiq.Game`.

## Run

Requires `numpy`, `scikit-learn`, `matplotlib`, and `pandas`. "Restart & Run All" takes about 20 seconds.

```bash
jupyter lab experiments/uncertainty_interactions/interaktionen_unsicherheitskarten.ipynb
```

The cells carry slideshow metadata, so the notebook can be presented with RISE or via
`jupyter nbconvert --to slides interaktionen_unsicherheitskarten.ipynb --post serve`.
