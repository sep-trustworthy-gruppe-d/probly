"""Draw the pipeline of the package as one figure for the presentation (figures/ablauf.png).

uv run python experiments/group-d_experiments/keith_03_vit_uncertainty/scripts/make_pipeline_figure.py
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

STEPS = [
    ("Bild", "Oxford-IIIT Pet\n224 x 224 Pixel, Dans Aufteilung", "#f2f2f2"),
    ("1. Modell", "ViT-S/16 + DINOv2-S/14\neingefroren, Tokens ausblendbar", "#dbe9f6"),
    ("2. Modell → probly", "Köpfe → Ensemble, MC Dropout,\ngemischtes Ensemble", "#e3f1df"),
    ("3. shapiq-Explainer", "16 SLIC-Superpixel als Spieler\nSpiele P, TU, AU, EU", "#fdebd3"),
    ("4. Erklärung von beidem", "Shapley-Karten + k-SII-Paare\nVorhersage neben Unsicherheit", "#f6dede"),
]


def main() -> None:
    """Draw the boxes and arrows and save the figure."""
    fig, ax = plt.subplots(figsize=(15, 3.2), layout="constrained")
    ax.set_xlim(0, len(STEPS) * 3)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for index, (title, text, colour) in enumerate(STEPS):
        x = index * 3 + 0.15
        ax.add_patch(FancyBboxPatch((x, 0.9), 2.5, 1.7, boxstyle="round,pad=0.05", facecolor=colour, edgecolor="0.3"))
        ax.text(x + 1.25, 2.25, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + 1.25, 1.45, text, ha="center", va="center", fontsize=9.5)
        if index < len(STEPS) - 1:
            ax.annotate("", xy=(x + 3.0, 1.75), xytext=(x + 2.65, 1.75), arrowprops={"arrowstyle": "-|>", "lw": 1.8})
    ax.text(
        len(STEPS) * 1.5,
        0.35,
        "5. ViT Shapley: ein lernender Erklärer schätzt alle vier Karten in einem Durchlauf "
        "(trainiert ohne echte Shapley-Werte, nur mit Spielwerten v(S))",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
    )
    out = Path(__file__).resolve().parents[1] / "figures" / "ablauf.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print("gespeichert:", out)


if __name__ == "__main__":
    main()
