"""
Heatmap of the cross-dataset AUC table
======================================

Reads cross_table.csv and produces a publication-quality heatmap
(scores × datasets) saved to figures/cross_auc_heatmap.png.

Run AFTER cross_table.py.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / "data" / "cross_table.csv"
OUT_PNG = ROOT / "figures" / "cross_auc_heatmap.png"
OUT_PDF = ROOT / "figures" / "cross_auc_heatmap.pdf"


# Curated rows: a clean subset of scores worth showing in the heatmap.
# We hide the duplicates (`*_all` variants on TriviaQA-num) and the
# answer-only sub-sequence variants (which only matter for §6.2).
DISPLAY_SCORES = [
    "prob_joint_seq",
    "prob_geo_mean_seq",
    "perplexity_seq",
    "p_min_seq",
    "epr",
    "wepr",
    "semantic_entropy",
]
# Pretty labels
LABELS = {
    "prob_joint_seq":        r"$P_{\mathrm{joint}}$",
    "prob_geo_mean_seq":     r"$P_{\mathrm{geo}}$",
    "perplexity_seq":        "Perplexité",
    "p_min_seq":             r"$p_{\min}$",
    "epr":                   "EPR",
    "wepr":                  "WEPR",
    "semantic_entropy":      "SE",
}

DATASET_ORDER = ["TriviaQA-num", "TriviaQA-WEPR", "GSM8K", "ELI5"]
DATASET_LABELS = {
    "TriviaQA-num":  "TriviaQA-num\n(n=3395)",
    "TriviaQA-WEPR": "TriviaQA-WEPR\n(n=73793)",
    "GSM8K":         "GSM8K\n(n=300)",
    "ELI5":          "ELI5\n(n=100)",
}


def load_auc_table(path: Path):
    """Return dict (score, dataset) -> auc."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            sc = row["score"]
            try:
                auc = float(row["auc"]) if row["auc"] else None
            except ValueError:
                auc = None
            if auc is not None:
                out[(sc, ds)] = auc
    return out


def main():
    auc = load_auc_table(IN_CSV)

    nrows = len(DISPLAY_SCORES)
    ncols = len(DATASET_ORDER)
    M = np.full((nrows, ncols), np.nan)
    for i, sc in enumerate(DISPLAY_SCORES):
        for j, ds in enumerate(DATASET_ORDER):
            v = auc.get((sc, ds))
            if v is not None:
                M[i, j] = v

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    # Use a diverging colormap centered at 0.5 (random).
    cmap = plt.get_cmap("RdYlGn")
    vmin, vmax = 0.5, 1.0
    masked = np.ma.masked_invalid(M)
    cmap.set_bad(color="#eeeeee")  # missing cells in light grey

    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")

    # Annotate cells
    for i in range(nrows):
        for j in range(ncols):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#888888")
            else:
                # Choose text color for readability
                color = "white" if (v < 0.62 or v > 0.92) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color=color)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER],
                       fontsize=10)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([LABELS[s] for s in DISPLAY_SCORES], fontsize=11)
    ax.set_title(
        "AUC de discrimination correct/incorrect par dataset",
        fontsize=12, pad=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("AUC", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Light grid between cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
