"""
ELI5 – Visualization and analysis of sentence-level uncertainty
================================================================
Generates figures from eli5_judged.jsonl:
  1. Sentence perplexity & entropy distributions
  2. Uncertainty vs sentence position
  3. Uncertainty vs judge accuracy score
  4. Per-question heatmaps (example sentences colored by uncertainty)
  5. Correlation scatter: overall uncertainty vs accuracy score

If eli5_semantic_entropy.jsonl exists, also generates:
  6. Semantic Entropy distribution
  7. Semantic Entropy vs accuracy score (boxplot + scatter)
  8. Semantic Entropy vs overall perplexity (scatter)
  9. Number of semantic clusters vs accuracy score

If eli5_selfcheck.jsonl exists, also generates:
 10. SelfCheck score distribution
 11. SelfCheck vs accuracy (boxplot + scatter)
 12. Confusion matrix: SE × SelfCheck
 13. Accuracy-coverage curves (all confidence signals)
 + Evaluation metrics table (Brier, ECE, AUROC, AUPRC)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

# ──────────────────────────── Config ────────────────────────────

JUDGED_JSONL = Path(__file__).resolve().parent / "eli5_judged.jsonl"
SE_JSONL = Path(__file__).resolve().parent / "eli5_semantic_entropy.jsonl"
SC_JSONL = Path(__file__).resolve().parent / "eli5_selfcheck.jsonl"
FIGS_DIR = Path(__file__).resolve().parent / "figures"


def load_results(path: Path) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" not in obj:
                results.append(obj)
    return results


# ──────────────────── Figure 1: Distributions ───────────────────

def plot_distributions(results: List[Dict], figs_dir: Path):
    """Boxplots of per-sentence perplexity and top-k entropy."""
    sent_ppls = [s["perplexity"] for r in results for s in r["sentences"]]
    sent_h = [s["topk_entropy"] for r in results for s in r["sentences"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Perplexity
    ax = axes[0]
    ax.hist(sent_ppls, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(sent_ppls), color="red", linestyle="--",
               label=f"median={np.median(sent_ppls):.2f}")
    ax.set_xlabel("Sentence Perplexity")
    ax.set_ylabel("Count")
    ax.set_title("Per-Sentence Perplexity Distribution")
    ax.legend()

    # Entropy
    ax = axes[1]
    ax.hist(sent_h, bins=30, color="#55A868", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(sent_h), color="red", linestyle="--",
               label=f"median={np.median(sent_h):.3f}")
    ax.set_xlabel("Sentence Top-k Entropy")
    ax.set_ylabel("Count")
    ax.set_title("Per-Sentence Top-k Entropy Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(figs_dir / "sentence_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved sentence_distributions.png")


# ──────────── Figure 2: Uncertainty vs sentence position ────────

def plot_uncertainty_vs_position(results: List[Dict], figs_dir: Path):
    """Line plot: average PPL and entropy by sentence index."""
    max_pos = max(r["num_sentences"] for r in results)
    max_pos = min(max_pos, 8)  # cap for readability

    ppls_by_pos = {i: [] for i in range(max_pos)}
    h_by_pos = {i: [] for i in range(max_pos)}

    for r in results:
        for s in r["sentences"]:
            si = s["sentence_index"]
            if si < max_pos:
                ppls_by_pos[si].append(s["perplexity"])
                h_by_pos[si].append(s["topk_entropy"])

    positions = list(range(max_pos))
    mean_ppls = [np.mean(ppls_by_pos[i]) if ppls_by_pos[i] else 0
                 for i in positions]
    std_ppls = [np.std(ppls_by_pos[i]) if ppls_by_pos[i] else 0
                for i in positions]
    mean_h = [np.mean(h_by_pos[i]) if h_by_pos[i] else 0
              for i in positions]
    std_h = [np.std(h_by_pos[i]) if h_by_pos[i] else 0
             for i in positions]
    counts = [len(ppls_by_pos[i]) for i in positions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Perplexity by position
    ax = axes[0]
    ax.errorbar(positions, mean_ppls, yerr=std_ppls, marker="o",
                capsize=4, color="#4C72B0", linewidth=2)
    ax.set_xlabel("Sentence Position")
    ax.set_ylabel("Mean Perplexity")
    ax.set_title("Perplexity by Sentence Position")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"S{i}\n(n={counts[i]})" for i in positions],
                       fontsize=8)

    # Entropy by position
    ax = axes[1]
    ax.errorbar(positions, mean_h, yerr=std_h, marker="o",
                capsize=4, color="#55A868", linewidth=2)
    ax.set_xlabel("Sentence Position")
    ax.set_ylabel("Mean Top-k Entropy")
    ax.set_title("Top-k Entropy by Sentence Position")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"S{i}\n(n={counts[i]})" for i in positions],
                       fontsize=8)

    fig.tight_layout()
    fig.savefig(figs_dir / "uncertainty_vs_position.png", dpi=150)
    plt.close(fig)
    print("  Saved uncertainty_vs_position.png")


# ──────── Figure 3: Uncertainty vs judge accuracy score ─────────

def plot_uncertainty_vs_accuracy(results: List[Dict], figs_dir: Path):
    """Boxplots of per-sentence PPL and entropy grouped by accuracy score."""
    # Group sentences by the accuracy score of their parent answer
    groups = {}  # accuracy_score -> {"ppls": [], "h": []}
    for r in results:
        score = (r.get("judge_scores") or {}).get("accuracy")
        if score is None:
            continue
        if score not in groups:
            groups[score] = {"ppls": [], "h": []}
        for s in r["sentences"]:
            groups[score]["ppls"].append(s["perplexity"])
            groups[score]["h"].append(s["topk_entropy"])

    if not groups:
        print("  No accuracy scores available, skipping.")
        return

    sorted_scores = sorted(groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Perplexity boxplot
    ax = axes[0]
    data_ppls = [groups[s]["ppls"] for s in sorted_scores]
    bp = ax.boxplot(data_ppls, tick_labels=[str(s) for s in sorted_scores],
                    patch_artist=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_scores)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Sentence Perplexity")
    ax.set_title("Perplexity by Accuracy Score")
    # Add counts
    for i, s in enumerate(sorted_scores):
        n = len(groups[s]["ppls"])
        ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center",
                va="bottom", fontsize=7, color="gray")

    # Entropy boxplot
    ax = axes[1]
    data_h = [groups[s]["h"] for s in sorted_scores]
    bp = ax.boxplot(data_h, tick_labels=[str(s) for s in sorted_scores],
                    patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Sentence Top-k Entropy")
    ax.set_title("Top-k Entropy by Accuracy Score")
    for i, s in enumerate(sorted_scores):
        n = len(groups[s]["h"])
        ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center",
                va="bottom", fontsize=7, color="gray")

    fig.tight_layout()
    fig.savefig(figs_dir / "uncertainty_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Saved uncertainty_vs_accuracy.png")


# ──────── Figure 4: Example sentence heatmaps ──────────────────

def plot_sentence_heatmaps(results: List[Dict], figs_dir: Path,
                           n_examples: int = 6):
    """Show a few example answers with sentences colored by perplexity."""
    # Pick diverse examples: sort by overall_perplexity, take spread
    sorted_results = sorted(results, key=lambda r: r["overall_perplexity"])
    indices = np.linspace(0, len(sorted_results) - 1, n_examples, dtype=int)
    examples = [sorted_results[i] for i in indices]

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 2.5 * n_examples))
    if n_examples == 1:
        axes = [axes]

    # Colormap: low PPL = green, high PPL = red
    all_ppls = [s["perplexity"] for r in results for s in r["sentences"]]
    vmin, vmax = np.percentile(all_ppls, 5), np.percentile(all_ppls, 95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r  # reversed: green=low, red=high

    for ax, r in zip(axes, examples):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Title: question (truncated)
        q = r["question"][:80] + ("..." if len(r["question"]) > 80 else "")
        accuracy = (r.get("judge_scores") or {}).get("accuracy", "?")
        ax.set_title(f"Q: {q}  [accuracy={accuracy}, PPL={r['overall_perplexity']:.2f}]",
                     fontsize=9, loc="left", pad=2)

        # Draw each sentence as a colored text block
        y = 0.6
        for s in r["sentences"]:
            ppl = s["perplexity"]
            color = cmap(norm(ppl))
            text = s["sentence_text"][:100]
            if len(s["sentence_text"]) > 100:
                text += "..."
            ax.text(0.02, y, text, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                              alpha=0.6, edgecolor="gray"),
                    wrap=True)
            # PPL label
            ax.text(0.98, y, f"PPL={ppl:.2f}\nH={s['topk_entropy']:.3f}",
                    fontsize=7, va="top", ha="right", color="gray")
            y -= 0.3

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="bottom", fraction=0.02,
                        pad=0.04, aspect=40)
    cbar.set_label("Sentence Perplexity", fontsize=9)

    fig.tight_layout()
    fig.savefig(figs_dir / "sentence_heatmaps.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved sentence_heatmaps.png")


# ──── Figure 5: Scatter overall uncertainty vs accuracy ─────────

def plot_scatter_uncertainty_accuracy(results: List[Dict], figs_dir: Path):
    """Scatter plot: overall perplexity/entropy vs accuracy score."""
    ppls = []
    ents = []
    scores = []
    for r in results:
        acc = (r.get("judge_scores") or {}).get("accuracy")
        if acc is None:
            continue
        ppls.append(r["overall_perplexity"])
        ents.append(r["overall_topk_entropy"])
        scores.append(acc)

    if not scores:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PPL vs accuracy
    ax = axes[0]
    jitter = np.random.uniform(-0.15, 0.15, len(scores))
    ax.scatter(np.array(scores) + jitter, ppls, alpha=0.5, s=30,
               c="#4C72B0", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Overall Perplexity")
    ax.set_title("Overall Perplexity vs Accuracy")
    # Correlation
    rho, pval = spearmanr(scores, ppls)
    ax.text(0.05, 0.95, f"Spearman r={rho:.3f}\np={pval:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Entropy vs accuracy
    ax = axes[1]
    ax.scatter(np.array(scores) + jitter, ents, alpha=0.5, s=30,
               c="#55A868", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Overall Top-k Entropy")
    ax.set_title("Overall Top-k Entropy vs Accuracy")
    rho, pval = spearmanr(scores, ents)
    ax.text(0.05, 0.95, f"Spearman r={rho:.3f}\np={pval:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(figs_dir / "scatter_uncertainty_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Saved scatter_uncertainty_accuracy.png")


# ──────── Figure 6: Semantic Entropy distribution ──────────────

def plot_se_distribution(se_data: List[Dict], figs_dir: Path):
    """Histogram of Semantic Entropy across all questions."""
    se_vals = [r["semantic_entropy"] for r in se_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(se_vals, bins=20, color="#C44E52", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(se_vals), color="navy", linestyle="--",
               label=f"median={np.median(se_vals):.3f}")
    ax.axvline(np.mean(se_vals), color="darkgreen", linestyle=":",
               label=f"mean={np.mean(se_vals):.3f}")
    ax.set_xlabel("Semantic Entropy")
    ax.set_ylabel("Count")
    ax.set_title("Semantic Entropy Distribution (K=5 responses per question)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(figs_dir / "semantic_entropy_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved semantic_entropy_distribution.png")


# ── Figure 7: Semantic Entropy vs accuracy (boxplot + scatter) ──

def plot_se_vs_accuracy(merged: List[Dict], figs_dir: Path):
    """Boxplot and scatter of SE grouped by accuracy score."""
    groups = {}
    se_vals = []
    acc_vals = []

    for r in merged:
        acc = r.get("accuracy")
        se = r.get("semantic_entropy")
        if acc is None or se is None:
            continue
        groups.setdefault(acc, []).append(se)
        se_vals.append(se)
        acc_vals.append(acc)

    if not groups:
        print("  No merged SE+accuracy data, skipping.")
        return

    sorted_scores = sorted(groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    data = [groups[s] for s in sorted_scores]
    bp = ax.boxplot(data, tick_labels=[str(s) for s in sorted_scores],
                    patch_artist=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_scores)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Semantic Entropy")
    ax.set_title("Semantic Entropy by Accuracy Score")
    for i, s in enumerate(sorted_scores):
        n = len(groups[s])
        ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center",
                va="bottom", fontsize=7, color="gray")

    # Scatter with correlation
    ax = axes[1]
    jitter = np.random.uniform(-0.15, 0.15, len(acc_vals))
    ax.scatter(np.array(acc_vals) + jitter, se_vals, alpha=0.5, s=30,
               c="#C44E52", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Semantic Entropy")
    ax.set_title("Semantic Entropy vs Accuracy (scatter)")
    if len(set(acc_vals)) > 1:
        rho, pval = spearmanr(acc_vals, se_vals)
        ax.text(0.05, 0.95, f"Spearman r={rho:.3f}\np={pval:.4f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(figs_dir / "se_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Saved se_vs_accuracy.png")


# ──── Figure 8: Semantic Entropy vs overall perplexity ─────────

def plot_se_vs_perplexity(merged: List[Dict], figs_dir: Path):
    """Scatter: SE vs overall perplexity to see if they're correlated."""
    se_vals = []
    ppl_vals = []

    for r in merged:
        se = r.get("semantic_entropy")
        ppl = r.get("overall_perplexity")
        if se is None or ppl is None:
            continue
        se_vals.append(se)
        ppl_vals.append(ppl)

    if len(se_vals) < 3:
        print("  Not enough data for SE vs PPL scatter, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ppl_vals, se_vals, alpha=0.6, s=40, c="#8172B2",
               edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Overall Perplexity")
    ax.set_ylabel("Semantic Entropy")
    ax.set_title("Semantic Entropy vs Overall Perplexity")

    rho, pval = spearmanr(ppl_vals, se_vals)
    ax.text(0.05, 0.95, f"Spearman r={rho:.3f}\np={pval:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(figs_dir / "se_vs_perplexity.png", dpi=150)
    plt.close(fig)
    print("  Saved se_vs_perplexity.png")


# ──── Figure 9: Num clusters vs accuracy score ─────────────────

def plot_clusters_vs_accuracy(merged: List[Dict], figs_dir: Path):
    """Bar plot: mean number of semantic clusters by accuracy score."""
    groups = {}
    for r in merged:
        acc = r.get("accuracy")
        nc = r.get("num_clusters")
        if acc is None or nc is None:
            continue
        groups.setdefault(acc, []).append(nc)

    if not groups:
        print("  No merged cluster+accuracy data, skipping.")
        return

    sorted_scores = sorted(groups.keys())
    means = [np.mean(groups[s]) for s in sorted_scores]
    stds = [np.std(groups[s]) for s in sorted_scores]
    counts = [len(groups[s]) for s in sorted_scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_scores)))
    bars = ax.bar([str(s) for s in sorted_scores], means, yerr=stds,
                  capsize=4, color=colors, edgecolor="gray", alpha=0.8)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("Mean Number of Clusters")
    ax.set_title("Semantic Clusters by Accuracy Score (K=5 responses)")

    for i, (bar, n) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(figs_dir / "clusters_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Saved clusters_vs_accuracy.png")


# ──────── Merge SE data with judged data ───────────────────────

def merge_se_with_judged(
    judged: List[Dict], se_data: List[Dict]
) -> List[Dict]:
    """
    Merge semantic entropy results with judged results by question_id.
    Returns list of dicts with fields from both sources.
    """
    se_by_qid = {r["question_id"]: r for r in se_data}

    merged = []
    for r in judged:
        qid = r.get("question_id")
        if qid and qid in se_by_qid:
            se = se_by_qid[qid]
            merged.append({
                "question_id": qid,
                "question": r["question"],
                "overall_perplexity": r.get("overall_perplexity"),
                "overall_topk_entropy": r.get("overall_topk_entropy"),
                "accuracy": (r.get("judge_scores") or {}).get("accuracy"),
                "simplicity": (r.get("judge_scores") or {}).get("simplicity"),
                "relevance": (r.get("judge_scores") or {}).get("relevance"),
                "semantic_entropy": se.get("semantic_entropy"),
                "num_clusters": se.get("num_clusters"),
            })

    return merged


# ──────── Evaluation metrics (from compare_confidence_mmlu.py) ──

def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.zeros(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def metric_brier(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((s - y) ** 2))


def metric_ece(y: np.ndarray, s: np.ndarray, bins: int = 10) -> float:
    s = np.clip(s, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not np.any(mask):
            continue
        conf = np.mean(s[mask])
        acc = np.mean(y[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def metric_auroc(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average_ties(s)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def metric_auprc(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def accuracy_coverage_rows(
    y: np.ndarray, s: np.ndarray
) -> List[Dict]:
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    n = len(y_sorted)
    rows = []
    for cov in np.arange(0.1, 1.05, 0.1):
        cov = round(cov, 1)
        k = max(1, int(np.ceil(cov * n)))
        acc = float(np.mean(y_sorted[:k]))
        rows.append({"coverage": cov, "k": k, "accuracy": acc})
    return rows


# ──── Figure 10: SelfCheck score distribution ─────────────────

def plot_selfcheck_distribution(sc_data: List[Dict], figs_dir: Path):
    """Histogram of SelfCheck scores across all questions."""
    scores = [r["selfcheck_score"] for r in sc_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=20, color="#DD8452", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(scores), color="navy", linestyle="--",
               label=f"median={np.median(scores):.3f}")
    ax.axvline(np.mean(scores), color="darkgreen", linestyle=":",
               label=f"mean={np.mean(scores):.3f}")
    ax.set_xlabel("SelfCheck Score (0=reliable, 1=contradicted)")
    ax.set_ylabel("Count")
    ax.set_title("SelfCheck Score Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(figs_dir / "selfcheck_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved selfcheck_distribution.png")


# ──── Figure 11: SelfCheck vs Accuracy ────────────────────────

def plot_selfcheck_vs_accuracy(merged: List[Dict], figs_dir: Path):
    """Boxplot and scatter of SelfCheck score grouped by accuracy."""
    groups = {}
    sc_vals = []
    acc_vals = []

    for r in merged:
        acc = r.get("accuracy")
        sc = r.get("selfcheck_score")
        if acc is None or sc is None:
            continue
        groups.setdefault(acc, []).append(sc)
        sc_vals.append(sc)
        acc_vals.append(acc)

    if not groups:
        print("  No merged SelfCheck+accuracy data, skipping.")
        return

    sorted_scores = sorted(groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    data = [groups[s] for s in sorted_scores]
    bp = ax.boxplot(data, tick_labels=[str(s) for s in sorted_scores],
                    patch_artist=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_scores)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("SelfCheck Score")
    ax.set_title("SelfCheck by Accuracy Score")
    for i, s in enumerate(sorted_scores):
        n = len(groups[s])
        ax.text(i + 1, ax.get_ylim()[0], f"n={n}", ha="center",
                va="bottom", fontsize=7, color="gray")

    # Scatter
    ax = axes[1]
    jitter = np.random.uniform(-0.15, 0.15, len(acc_vals))
    ax.scatter(np.array(acc_vals) + jitter, sc_vals, alpha=0.5, s=30,
               c="#DD8452", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Judge Accuracy Score")
    ax.set_ylabel("SelfCheck Score")
    ax.set_title("SelfCheck vs Accuracy (scatter)")
    if len(set(acc_vals)) > 1:
        rho, pval = spearmanr(acc_vals, sc_vals)
        ax.text(0.05, 0.95, f"Spearman r={rho:.3f}\np={pval:.4f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(figs_dir / "selfcheck_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print("  Saved selfcheck_vs_accuracy.png")


# ──── Figure 12: SE × SelfCheck confusion matrix ─────────────

def plot_se_selfcheck_matrix(merged: List[Dict], figs_dir: Path):
    """
    2×2 heatmap: (SE=0/SE>0) × (SelfCheck<0.3/SelfCheck>=0.3)
    colored by mean accuracy.
    """
    cells = {(r, c): [] for r in range(2) for c in range(2)}

    for m in merged:
        se = m.get("semantic_entropy")
        sc = m.get("selfcheck_score")
        acc = m.get("accuracy")
        if se is None or sc is None or acc is None:
            continue
        row = 0 if se == 0 else 1       # SE=0 vs SE>0
        col = 0 if sc < 0.3 else 1      # SelfCheck low vs high
        cells[(row, col)].append(acc)

    matrix = np.full((2, 2), np.nan)
    annot = [[" "] * 2 for _ in range(2)]

    for (r, c), vals in cells.items():
        if vals:
            mean_acc = np.mean(vals)
            matrix[r, c] = mean_acc
            annot[r][c] = f"acc={mean_acc:.2f}\nn={len(vals)}"

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SelfCheck < 0.3\n(reliable)", "SelfCheck >= 0.3\n(unreliable)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SE = 0\n(confident)", "SE > 0\n(uncertain)"])
    ax.set_title("Mean Accuracy: SE × SelfCheck")

    for r in range(2):
        for c in range(2):
            ax.text(c, r, annot[r][c], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if not np.isnan(matrix[r, c]) else "gray")

    fig.colorbar(im, ax=ax, label="Mean Accuracy Score (1-5)")
    fig.tight_layout()
    fig.savefig(figs_dir / "se_selfcheck_matrix.png", dpi=150)
    plt.close(fig)
    print("  Saved se_selfcheck_matrix.png")


# ──── Figure 13: Accuracy-coverage curves ─────────────────────

def plot_accuracy_coverage(merged: List[Dict], figs_dir: Path):
    """
    For each confidence signal, plot accuracy vs coverage.
    Higher accuracy at lower coverage = better confidence signal.
    """
    # Build arrays
    records = []
    for m in merged:
        acc = m.get("accuracy")
        if acc is None:
            continue
        records.append(m)

    if len(records) < 10:
        print("  Not enough data for coverage curves, skipping.")
        return

    y = np.array([1.0 if r["accuracy"] >= 4 else 0.0 for r in records])

    # Build confidence signals (higher = more confident)
    signals = {}

    # Perplexity (invert: high PPL → low confidence)
    ppls = [r.get("overall_perplexity") for r in records]
    if all(p is not None for p in ppls):
        ppls = np.array(ppls)
        ppl_min, ppl_max = ppls.min(), ppls.max()
        if ppl_max > ppl_min:
            signals["c_ppl"] = 1.0 - (ppls - ppl_min) / (ppl_max - ppl_min)

    # Top-k entropy (invert)
    ents = [r.get("overall_topk_entropy") for r in records]
    if all(e is not None for e in ents):
        ents = np.array(ents)
        e_min, e_max = ents.min(), ents.max()
        if e_max > e_min:
            signals["c_entropy"] = 1.0 - (ents - e_min) / (e_max - e_min)

    # Semantic Entropy (invert)
    ses = [r.get("semantic_entropy") for r in records]
    if all(s is not None for s in ses):
        ses = np.array(ses)
        se_min, se_max = ses.min(), ses.max()
        if se_max > se_min:
            signals["c_se"] = 1.0 - (ses - se_min) / (se_max - se_min)

    # SelfCheck (invert: high score → low confidence)
    scs = [r.get("selfcheck_score") for r in records]
    if all(s is not None for s in scs):
        scs = np.array(scs)
        sc_min, sc_max = scs.min(), scs.max()
        if sc_max > sc_min:
            signals["c_selfcheck"] = 1.0 - (scs - sc_min) / (sc_max - sc_min)

    # Combined = mean of all available signals
    if len(signals) >= 2:
        combined = np.mean(
            [signals[k] for k in signals], axis=0
        )
        signals["c_combined"] = combined

    if not signals:
        print("  No confidence signals available, skipping coverage curves.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_map = {
        "c_ppl": "#4C72B0",
        "c_entropy": "#55A868",
        "c_se": "#C44E52",
        "c_selfcheck": "#DD8452",
        "c_combined": "#8172B2",
    }
    labels_map = {
        "c_ppl": "Perplexity",
        "c_entropy": "Top-k Entropy",
        "c_se": "Semantic Entropy",
        "c_selfcheck": "SelfCheck",
        "c_combined": "Combined",
    }

    for sig_name, sig_vals in signals.items():
        rows = accuracy_coverage_rows(y, sig_vals)
        coverages = [r["coverage"] for r in rows]
        accuracies = [r["accuracy"] for r in rows]
        color = colors_map.get(sig_name, "gray")
        label = labels_map.get(sig_name, sig_name)
        lw = 3 if sig_name == "c_combined" else 1.5
        ls = "-" if sig_name == "c_combined" else "--"
        ax.plot(coverages, accuracies, marker="o", markersize=4,
                color=color, linewidth=lw, linestyle=ls, label=label)

    # Baseline
    base_acc = float(np.mean(y))
    ax.axhline(base_acc, color="gray", linestyle=":", linewidth=1,
               label=f"Baseline ({base_acc:.2f})")

    ax.set_xlabel("Coverage (fraction of questions kept)")
    ax.set_ylabel("Accuracy (acc >= 4)")
    ax.set_title("Accuracy vs Coverage by Confidence Signal")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figs_dir / "accuracy_coverage.png", dpi=150)
    plt.close(fig)
    print("  Saved accuracy_coverage.png")

    # ---- Print metrics table ----
    print(f"\n  {'Signal':<16} {'Brier':>7} {'ECE':>7} {'AUROC':>7} {'AUPRC':>7}")
    print(f"  {'-' * 46}")
    for sig_name, sig_vals in signals.items():
        label = labels_map.get(sig_name, sig_name)
        brier = metric_brier(y, sig_vals)
        ece = metric_ece(y, sig_vals)
        auroc = metric_auroc(y, sig_vals)
        auprc = metric_auprc(y, sig_vals)
        print(f"  {label:<16} {brier:7.4f} {ece:7.4f} {auroc:7.4f} {auprc:7.4f}")


# ──── Merge SelfCheck with judged+SE data ─────────────────────

def merge_all_data(
    judged: List[Dict],
    se_data: List[Dict],
    sc_data: List[Dict],
) -> List[Dict]:
    """
    Merge judged + SE + SelfCheck data by question_id.
    Returns list of dicts with fields from all three sources.
    """
    se_by_qid = {r["question_id"]: r for r in se_data}
    sc_by_qid = {r["question_id"]: r for r in sc_data}

    merged = []
    for r in judged:
        qid = r.get("question_id")
        if not qid:
            continue

        entry = {
            "question_id": qid,
            "question": r["question"],
            "overall_perplexity": r.get("overall_perplexity"),
            "overall_topk_entropy": r.get("overall_topk_entropy"),
            "accuracy": (r.get("judge_scores") or {}).get("accuracy"),
            "simplicity": (r.get("judge_scores") or {}).get("simplicity"),
            "relevance": (r.get("judge_scores") or {}).get("relevance"),
        }

        se = se_by_qid.get(qid)
        if se:
            entry["semantic_entropy"] = se.get("semantic_entropy")
            entry["num_clusters"] = se.get("num_clusters")

        sc = sc_by_qid.get(qid)
        if sc:
            entry["selfcheck_score"] = sc.get("selfcheck_score")

        merged.append(entry)

    return merged


# ──────────────────────────── Main ──────────────────────────────

def main():
    results = load_results(JUDGED_JSONL)
    print(f"Loaded {len(results)} results from {JUDGED_JSONL}")

    if not results:
        print("No results to analyze.")
        return

    FIGS_DIR.mkdir(exist_ok=True)

    print("\nGenerating figures...")
    plot_distributions(results, FIGS_DIR)
    plot_uncertainty_vs_position(results, FIGS_DIR)
    plot_uncertainty_vs_accuracy(results, FIGS_DIR)
    plot_sentence_heatmaps(results, FIGS_DIR)
    plot_scatter_uncertainty_accuracy(results, FIGS_DIR)

    # Print summary stats
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    all_ppls = [s["perplexity"] for r in results for s in r["sentences"]]
    all_h = [s["topk_entropy"] for r in results for s in r["sentences"]]
    print(f"  Total sentences: {len(all_ppls)}")
    print(f"  Per-sentence PPL:     mean={np.mean(all_ppls):.2f}  "
          f"median={np.median(all_ppls):.2f}  std={np.std(all_ppls):.2f}")
    print(f"  Per-sentence entropy: mean={np.mean(all_h):.3f}  "
          f"median={np.median(all_h):.3f}  std={np.std(all_h):.3f}")

    # Accuracy correlation
    scores = []
    ppls = []
    for r in results:
        acc = (r.get("judge_scores") or {}).get("accuracy")
        if acc is not None:
            scores.append(acc)
            ppls.append(r["overall_perplexity"])
    if len(set(scores)) > 1:
        rho, pval = spearmanr(scores, ppls)
        print(f"\n  Spearman correlation (accuracy vs overall PPL): "
              f"r={rho:.3f}, p={pval:.4f}")

    # ──── Semantic Entropy figures (if data available) ────
    if SE_JSONL.exists():
        se_data = load_results(SE_JSONL)
        print(f"\nLoaded {len(se_data)} SE results from {SE_JSONL}")

        if se_data:
            print("\nGenerating Semantic Entropy figures...")
            plot_se_distribution(se_data, FIGS_DIR)

            merged = merge_se_with_judged(results, se_data)
            print(f"Merged {len(merged)} questions (SE + judged)")

            if merged:
                plot_se_vs_accuracy(merged, FIGS_DIR)
                plot_se_vs_perplexity(merged, FIGS_DIR)
                plot_clusters_vs_accuracy(merged, FIGS_DIR)

                # SE summary
                se_vals = [m["semantic_entropy"] for m in merged
                           if m["semantic_entropy"] is not None]
                if se_vals:
                    print(f"\n  Semantic Entropy: mean={np.mean(se_vals):.4f}  "
                          f"median={np.median(se_vals):.4f}")
                acc_vals = [m["accuracy"] for m in merged
                            if m["accuracy"] is not None
                            and m["semantic_entropy"] is not None]
                se_for_corr = [m["semantic_entropy"] for m in merged
                               if m["accuracy"] is not None
                               and m["semantic_entropy"] is not None]
                if len(set(acc_vals)) > 1:
                    rho, pval = spearmanr(acc_vals, se_for_corr)
                    print(f"  Spearman correlation (accuracy vs SE): "
                          f"r={rho:.3f}, p={pval:.4f}")
    else:
        print(f"\n  {SE_JSONL.name} not found — skipping SE figures.")
        print("  Run semantic_entropy.py first to generate SE data.")

    # ──── SelfCheck figures (if data available) ────
    if SC_JSONL.exists():
        sc_data = load_results(SC_JSONL)
        print(f"\nLoaded {len(sc_data)} SelfCheck results from {SC_JSONL}")

        if sc_data:
            print("\nGenerating SelfCheck figures...")
            plot_selfcheck_distribution(sc_data, FIGS_DIR)

            # Merge all three data sources
            se_data_for_merge = load_results(SE_JSONL) if SE_JSONL.exists() else []
            merged_all = merge_all_data(results, se_data_for_merge, sc_data)
            print(f"Merged {len(merged_all)} questions (judged + SE + SelfCheck)")

            if merged_all:
                plot_selfcheck_vs_accuracy(merged_all, FIGS_DIR)

                # Only plot SE×SelfCheck matrix if we have SE data
                has_se = any(m.get("semantic_entropy") is not None
                             for m in merged_all)
                if has_se:
                    plot_se_selfcheck_matrix(merged_all, FIGS_DIR)

                # Accuracy-coverage + metrics table
                print("\nGenerating accuracy-coverage curves + metrics...")
                plot_accuracy_coverage(merged_all, FIGS_DIR)

                # SelfCheck summary
                sc_scores = [m["selfcheck_score"] for m in merged_all
                             if m.get("selfcheck_score") is not None]
                if sc_scores:
                    print(f"\n  SelfCheck: mean={np.mean(sc_scores):.4f}  "
                          f"median={np.median(sc_scores):.4f}")
                acc_vals = [m["accuracy"] for m in merged_all
                            if m.get("accuracy") is not None
                            and m.get("selfcheck_score") is not None]
                sc_for_corr = [m["selfcheck_score"] for m in merged_all
                               if m.get("accuracy") is not None
                               and m.get("selfcheck_score") is not None]
                if len(set(acc_vals)) > 1:
                    rho, pval = spearmanr(acc_vals, sc_for_corr)
                    print(f"  Spearman correlation (accuracy vs SelfCheck): "
                          f"r={rho:.3f}, p={pval:.4f}")
    else:
        print(f"\n  {SC_JSONL.name} not found — skipping SelfCheck figures.")
        print("  Run selfcheck_eli5.py first to generate SelfCheck data.")

    print()


if __name__ == "__main__":
    main()
