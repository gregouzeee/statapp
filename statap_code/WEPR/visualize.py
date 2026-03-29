"""
WEPR Visualization — Uncertainty Heatmap
=========================================
Generates heatmap visualizations of token/word-level uncertainty
using WEPR scores, EPR, and cosine importance.

Can be used on:
    - A single text from the judged dataset
    - A new query (generates answer via Gemini API)

Usage:
    python visualize.py --index 5          # visualize 5th sample from dataset
    python visualize.py --query "What is the capital of France?"
    python visualize.py --all --max 20     # batch visualize first 20 samples
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

from wepr import (
    epr, per_token_entropy, wepr_token_scores, wepr_word_scores,
    cosine_leave_one_out, cosine_word_leave_one_out,
    group_tokens_into_words,
)


# ──────────────────────────── Config ────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
JUDGED_JSONL = DATA_DIR / "triviaqa_judged.jsonl"
MODEL_JSON = DATA_DIR / "wepr_model.json"
FIGS_DIR = Path(__file__).resolve().parent / "figures"


# ──────────────────────── Load model ────────────────────────

def load_model(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: Path) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj or "token_data" not in obj:
                continue
            td = obj["token_data"]
            if td and "top_k" in td[0]:
                results.append(obj)
    return results


# ──────────────────────── Heatmap: token level ────────────────────────

def plot_token_heatmap(token_data: List[Dict], K: int, beta: np.ndarray,
                       title: str = "", save_path: Optional[Path] = None):
    """
    Plot a heatmap of token-level uncertainty scores.

    Three rows:
        1. Entropy H_K(j) per token
        2. WEPR risk sigma(S_beta(j)) per token
        3. Cosine leave-one-out importance per token
    """
    # Compute scores
    entropies = per_token_entropy(token_data, K)
    wepr_scores = wepr_token_scores(token_data, K, beta)
    cosine_scores = cosine_leave_one_out(token_data)

    tokens = [td["token"] for td in token_data]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return

    # Prepare data arrays
    entropy_vals = entropies
    risk_vals = np.array([ws["risk"] for ws in wepr_scores])
    cosine_vals = np.array([cs["importance"] for cs in cosine_scores])

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(max(n_tokens * 0.6, 10), 6),
                             gridspec_kw={"height_ratios": [1, 1, 1]})

    cmap = plt.cm.RdYlGn_r  # Red = high uncertainty, Green = low

    for ax, vals, label in zip(axes,
                                [entropy_vals, risk_vals, cosine_vals],
                                ["Entropy H_K", "WEPR risk", "Cosine LOO"]):
        # Normalize to [0, 1] for coloring
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-10:
            norm_vals = np.zeros_like(vals)
        else:
            norm_vals = (vals - vmin) / (vmax - vmin)

        for j in range(n_tokens):
            color = cmap(norm_vals[j])
            ax.add_patch(FancyBboxPatch(
                (j, 0), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="gray",
                linewidth=0.5,
            ))
            # Token text
            tok_display = tokens[j].replace(" ", "_") if tokens[j].startswith(" ") else tokens[j]
            ax.text(j + 0.45, 0.4, tok_display,
                    ha="center", va="center", fontsize=7,
                    rotation=45 if len(tok_display) > 4 else 0)
            # Score below
            ax.text(j + 0.45, -0.15, f"{vals[j]:.2f}",
                    ha="center", va="top", fontsize=5, color="gray")

        ax.set_xlim(-0.2, n_tokens + 0.2)
        ax.set_ylim(-0.3, 1.0)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────── Heatmap: word level ────────────────────────

def plot_word_heatmap(token_data: List[Dict], K: int, beta: np.ndarray,
                      title: str = "", save_path: Optional[Path] = None):
    """
    Plot word-level uncertainty heatmap.

    Two rows:
        1. WEPR word risk (max risk among sub-tokens)
        2. Cosine word leave-one-out importance
    """
    wepr_words = wepr_word_scores(token_data, K, beta)
    cosine_words = cosine_word_leave_one_out(token_data)

    if not wepr_words:
        return

    words_text = [w["word"] for w in wepr_words]
    n_words = len(words_text)

    wepr_risk = np.array([w["max_risk"] for w in wepr_words])

    # Align cosine with WEPR words
    cosine_dict = {w["word"]: w["importance"] for w in cosine_words}
    cosine_vals = np.array([cosine_dict.get(w, 0.0) for w in words_text])

    fig, axes = plt.subplots(2, 1, figsize=(max(n_words * 1.2, 10), 4),
                             gridspec_kw={"height_ratios": [1, 1]})

    cmap = plt.cm.RdYlGn_r

    for ax, vals, label in zip(axes,
                                [wepr_risk, cosine_vals],
                                ["WEPR word risk", "Cosine word LOO"]):
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-10:
            norm_vals = np.zeros_like(vals)
        else:
            norm_vals = (vals - vmin) / (vmax - vmin)

        for j in range(n_words):
            color = cmap(norm_vals[j])
            ax.add_patch(FancyBboxPatch(
                (j, 0), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="gray",
                linewidth=0.5,
            ))
            ax.text(j + 0.45, 0.4, words_text[j],
                    ha="center", va="center", fontsize=8,
                    fontweight="bold" if norm_vals[j] > 0.7 else "normal")
            ax.text(j + 0.45, -0.15, f"{vals[j]:.3f}",
                    ha="center", va="top", fontsize=6, color="gray")

        ax.set_xlim(-0.2, n_words + 0.2)
        ax.set_ylim(-0.3, 1.0)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────── Inline colored text ────────────────────────

def print_colored_text(token_data: List[Dict], K: int, beta: np.ndarray):
    """
    Print the generated text with ANSI colors indicating uncertainty.
    Green = confident, Yellow = moderate, Red = high risk.
    """
    scores = wepr_token_scores(token_data, K, beta)

    risks = [s["risk"] for s in scores]
    max_risk = max(risks) if risks else 1.0

    for s in scores:
        r = s["risk"]
        # Normalize to [0, 1]
        norm_r = r / max_risk if max_risk > 0 else 0

        if norm_r < 0.3:
            color = "\033[92m"  # green
        elif norm_r < 0.6:
            color = "\033[93m"  # yellow
        else:
            color = "\033[91m"  # red

        print(f"{color}{s['token']}\033[0m", end="")
    print()


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WEPR visualization")
    parser.add_argument("--index", type=int, default=None,
                        help="Index of sample to visualize from dataset")
    parser.add_argument("--all", action="store_true",
                        help="Batch visualize multiple samples")
    parser.add_argument("--max", type=int, default=10,
                        help="Max samples for --all mode")
    parser.add_argument("--K", type=int, default=None,
                        help="Top-K (defaults to model's K)")
    args = parser.parse_args()

    # Load model
    if not MODEL_JSON.exists():
        print(f"Model not found: {MODEL_JSON}")
        print("Run train_wepr.py first, or use EPR mode (no learned weights).")
        print("Using uniform weights (EPR equivalent)...")
        K = args.K or 10
        beta = np.ones(K + 1)
        beta[0] = 0.0
    else:
        model = load_model(MODEL_JSON)
        K = args.K or model["K"]
        beta = np.array(model["beta"])

    # Load data
    data = load_data(JUDGED_JSONL)
    if not data:
        print(f"No data found in {JUDGED_JSONL}")
        return

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.index is not None:
        # Single sample
        if args.index >= len(data):
            print(f"Index {args.index} out of range (max {len(data)-1})")
            return

        item = data[args.index]
        td = item["token_data"]
        question = item.get("question", "")
        answer = item.get("full_answer", "")
        correct = item.get("judge_correct", None)
        status = "CORRECT" if correct else "HALLUCINATED" if correct is False else "UNKNOWN"

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Judge: {status}")
        print(f"EPR: {epr(td, K):.4f}")
        print()
        print("Colored text (green=confident, red=risky):")
        print_colored_text(td, K, beta)
        print()

        title = f"Q: {question[:80]}...\nA: {answer[:80]}... [{status}]"
        plot_token_heatmap(td, K, beta, title=title,
                           save_path=FIGS_DIR / f"heatmap_token_{args.index}.png")
        plot_word_heatmap(td, K, beta, title=title,
                          save_path=FIGS_DIR / f"heatmap_word_{args.index}.png")
        print(f"Saved to {FIGS_DIR}/")

    elif args.all:
        n = min(args.max, len(data))
        print(f"Generating heatmaps for {n} samples...")

        for i in range(n):
            item = data[i]
            td = item["token_data"]
            question = item.get("question", "")[:60]
            answer = item.get("full_answer", "")[:60]
            correct = item.get("judge_correct", None)
            status = "OK" if correct else "HALLU" if correct is False else "?"

            title = f"[{i}] Q: {question}...\nA: {answer}... [{status}]"
            plot_word_heatmap(td, K, beta, title=title,
                              save_path=FIGS_DIR / f"heatmap_word_{i}.png")

        print(f"Saved {n} heatmaps to {FIGS_DIR}/")

    else:
        # Default: show first sample
        print("No --index or --all specified. Showing first sample.")
        item = data[0]
        td = item["token_data"]
        print(f"Q: {item.get('question', '')}")
        print(f"A: {item.get('full_answer', '')}")
        print()
        print_colored_text(td, K, beta)


if __name__ == "__main__":
    main()
