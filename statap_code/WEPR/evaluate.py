"""
Evaluate WEPR — standalone and combined with other methods
==========================================================
Compares WEPR with EPR, cosine leave-one-out, and combinations.

Uses bootstrapping (1000 iterations) like the original paper.

Reads:
    - data/triviaqa_judged.jsonl
    - data/wepr_model.json

Outputs:
    - figures/roc_curves.png
    - figures/score_distributions.png
    - figures/bootstrap_comparison.png
    - figures/pr_curves.png
    - data/evaluation_results.json

Usage:
    python evaluate.py [--K 10] [--n_bootstrap 1000]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from wepr import (
    epr, wepr_features, wepr_score, wepr_sequence_confidence,
    cosine_leave_one_out, cosine_word_leave_one_out,
    per_token_entropy,
)
from train_wepr import load_judged_data


# ──────────────────────────── Config ────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
JUDGED_JSONL = DATA_DIR / "triviaqa_judged.jsonl"
MODEL_JSON = DATA_DIR / "wepr_model.json"
FIGS_DIR = Path(__file__).resolve().parent / "figures"


# ──────────────── Feature extraction ─────────────────

def extract_all_features(data: List[Dict], K: int) -> Dict[str, np.ndarray]:
    n = len(data)

    features = {
        "epr": np.zeros(n),
        "mean_logprob": np.zeros(n),
        "min_logprob": np.zeros(n),
        "sentence_perplexity": np.zeros(n),
    }

    wepr_feat_list = []

    for i, item in enumerate(data):
        td = item["token_data"]

        features["epr"][i] = epr(td, K)

        logprobs = [t["log_prob"] for t in td]
        features["mean_logprob"][i] = np.mean(logprobs) if logprobs else 0.0
        features["min_logprob"][i] = np.min(logprobs) if logprobs else 0.0

        avg_lp = np.mean(logprobs) if logprobs else 0.0
        features["sentence_perplexity"][i] = np.exp(-avg_lp)

        wf = wepr_features(td, K)
        wepr_feat_list.append(wf["feature_vector"])

    features["wepr_features"] = np.array(wepr_feat_list)

    return features


# ──────────────── Bootstrap evaluation ─────────────────

def bootstrap_single_score(scores: np.ndarray, y: np.ndarray,
                           higher_is_correct: bool, n_bootstrap: int,
                           seed: int = 42) -> Dict:
    """Bootstrap evaluation for a single scalar score (no training)."""
    rng = np.random.RandomState(seed)
    if not higher_is_correct:
        scores = -scores

    aucs = []
    for _ in tqdm(range(n_bootstrap), desc="  Bootstrap", unit="iter", leave=False):
        idx = rng.choice(len(y), size=len(y), replace=True)
        y_b = y[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(roc_auc_score(y_b, s_b))

    return {
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
        "ci_lower": float(np.percentile(aucs, 2.5)),
        "ci_upper": float(np.percentile(aucs, 97.5)),
        "all_aucs": aucs,
    }


def bootstrap_learned(X: np.ndarray, y: np.ndarray,
                      n_bootstrap: int, seed: int = 42,
                      subsample: int = 10000) -> Dict:
    """Bootstrap evaluation for a learned method (train on bootstrap, test on OOB).
    Subsamples to `subsample` rows for speed when dataset is large."""
    rng = np.random.RandomState(seed)

    # Subsample for speed if dataset is large
    n = len(y)
    if n > subsample:
        sub_idx = rng.choice(n, size=subsample, replace=False)
        X = X[sub_idx]
        y = y[sub_idx]
        n = subsample

    aucs = []

    for _ in tqdm(range(n_bootstrap), desc="  Bootstrap", unit="iter", leave=False):
        idx = rng.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) < 10 or len(np.unique(y[oob_idx])) < 2:
            continue
        if len(np.unique(y[idx])) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[idx])
        X_test = scaler.transform(X[oob_idx])

        clf = LogisticRegression(C=1e10, max_iter=500, solver="lbfgs",
                                 random_state=seed)
        clf.fit(X_train, y[idx])

        y_prob = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y[oob_idx], y_prob))

    return {
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
        "ci_lower": float(np.percentile(aucs, 2.5)),
        "ci_upper": float(np.percentile(aucs, 97.5)),
        "all_aucs": aucs,
    }


# ──────────────── Plotting ─────────────────

def plot_roc_curves(features: Dict, y: np.ndarray, K: int, save_path: Path):
    """Plot ROC curves for all single-score methods + WEPR."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Single-score methods
    methods = [
        ("EPR", -features["epr"]),
        ("Mean log-prob", features["mean_logprob"]),
        ("Min log-prob", features["min_logprob"]),
        ("Perplexity", -features["sentence_perplexity"]),
    ]

    for name, scores in methods:
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", alpha=0.8)

    # WEPR (train on all, just for the curve shape)
    X_wepr = features["wepr_features"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wepr)
    clf = LogisticRegression(C=np.inf, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    ax.plot(fpr, tpr, label=f"WEPR (AUC={auc:.3f})", linewidth=2, color="black")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Hallucination Detection", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(features: Dict, y: np.ndarray, K: int, save_path: Path):
    """Plot Precision-Recall curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    methods = [
        ("EPR", -features["epr"]),
        ("Mean log-prob", features["mean_logprob"]),
        ("Min log-prob", features["min_logprob"]),
    ]

    for name, scores in methods:
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", alpha=0.8)

    # WEPR
    X_wepr = features["wepr_features"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_wepr)
    clf = LogisticRegression(C=np.inf, max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    ap = average_precision_score(y, y_prob)
    ax.plot(recall, precision, label=f"WEPR (AP={ap:.3f})", linewidth=2, color="black")

    baseline = y.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Hallucination Detection", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(features: Dict, y: np.ndarray, save_path: Path):
    """Plot score distributions for correct vs hallucinated."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = [
        ("EPR (entropy)", features["epr"], axes[0, 0]),
        ("Mean log-prob", features["mean_logprob"], axes[0, 1]),
        ("Min log-prob", features["min_logprob"], axes[1, 0]),
        ("Perplexity", features["sentence_perplexity"], axes[1, 1]),
    ]

    for name, scores, ax in methods:
        correct = scores[y == 1]
        hallu = scores[y == 0]
        ax.hist(correct, bins=50, alpha=0.6, label=f"Correct (n={len(correct)})",
                density=True, color="green")
        ax.hist(hallu, bins=50, alpha=0.6, label=f"Hallucinated (n={len(hallu)})",
                density=True, color="red")
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Score Distributions: Correct vs Hallucinated", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_comparison(results: Dict, save_path: Path):
    """Plot bootstrap AUC distributions as box plots."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    names = []
    all_aucs = []

    for name, data in results.items():
        names.append(name)
        all_aucs.append(data["all_aucs"])

    bp = ax.boxplot(all_aucs, labels=names, patch_artist=True, vert=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    for i, aucs in enumerate(all_aucs):
        ax.plot(i + 1, np.mean(aucs), "ko", markersize=5)
        ax.annotate(f"{np.mean(aucs):.3f}",
                    xy=(i + 1, np.mean(aucs)),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Bootstrap ROC-AUC Comparison (1000 iterations)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── Main ─────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate WEPR and combinations")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--input", type=str, default=str(JUDGED_JSONL))
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)

    # Load data
    data = load_judged_data(input_path)
    print(f"Loaded {len(data)} judged samples")

    y = np.array([1 if d["judge_correct"] else 0 for d in data])
    n_correct = y.sum()
    n_hallu = len(y) - n_correct
    print(f"  Correct: {n_correct} ({n_correct/len(y):.1%})")
    print(f"  Hallucinated: {n_hallu} ({n_hallu/len(y):.1%})")

    # Extract features
    print(f"\nExtracting features (K={args.K})...")
    features = extract_all_features(data, args.K)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ──── Bootstrap evaluation ────
    print(f"\nBootstrap evaluation ({args.n_bootstrap} iterations)...")

    bootstrap_results = {}

    # Single-score methods
    print("  EPR...")
    bootstrap_results["EPR"] = bootstrap_single_score(
        features["epr"], y, higher_is_correct=False, n_bootstrap=args.n_bootstrap)

    print("  Mean log-prob...")
    bootstrap_results["Mean log-prob"] = bootstrap_single_score(
        features["mean_logprob"], y, higher_is_correct=True, n_bootstrap=args.n_bootstrap)

    print("  Min log-prob...")
    bootstrap_results["Min log-prob"] = bootstrap_single_score(
        features["min_logprob"], y, higher_is_correct=True, n_bootstrap=args.n_bootstrap)

    print("  Perplexity...")
    bootstrap_results["Perplexity"] = bootstrap_single_score(
        features["sentence_perplexity"], y, higher_is_correct=False, n_bootstrap=args.n_bootstrap)

    # Learned methods
    print("  WEPR (learned, bootstrap train/OOB test)...")
    bootstrap_results["WEPR"] = bootstrap_learned(
        features["wepr_features"], y, n_bootstrap=args.n_bootstrap)

    # Print results
    print(f"\n{'='*75}")
    print(f" BOOTSTRAP RESULTS ({args.n_bootstrap} iterations)")
    print(f"{'='*75}")
    print(f"\n  {'Method':<20s}  {'AUC':>8s}  {'Std':>8s}  {'95% CI':>18s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*18}")
    for name, res in bootstrap_results.items():
        ci = f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
        print(f"  {name:<20s}  {res['mean']:>8.4f}  {res['std']:>8.4f}  {ci:>18s}")

    # ──── Generate plots ────
    print("\nGenerating figures...")

    print("  ROC curves...")
    plot_roc_curves(features, y, args.K, FIGS_DIR / "roc_curves.png")

    print("  PR curves...")
    plot_pr_curves(features, y, args.K, FIGS_DIR / "pr_curves.png")

    print("  Score distributions...")
    plot_score_distributions(features, y, FIGS_DIR / "score_distributions.png")

    print("  Bootstrap comparison...")
    plot_bootstrap_comparison(bootstrap_results, FIGS_DIR / "bootstrap_comparison.png")

    # ──── Save results ────
    save_results = {
        "K": args.K,
        "n_samples": len(y),
        "n_correct": int(n_correct),
        "n_hallucinated": int(n_hallu),
        "n_bootstrap": args.n_bootstrap,
        "bootstrap": {
            name: {k: v for k, v in res.items() if k != "all_aucs"}
            for name, res in bootstrap_results.items()
        },
    }
    output_path = DATA_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Figures saved to {FIGS_DIR}/")
    print("  - roc_curves.png")
    print("  - pr_curves.png")
    print("  - score_distributions.png")
    print("  - bootstrap_comparison.png")


if __name__ == "__main__":
    main()
