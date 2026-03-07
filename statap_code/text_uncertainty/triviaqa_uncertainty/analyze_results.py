"""
Analyse des résultats TriviaQA – Comparaison Perplexity vs Cosine Similarity
=============================================================================
Charge le fichier JSONL produit par main_triviaqa.py et génère :
  - Tableau récapitulatif par question
  - Visualisations comparatives des deux méthodes
  - Analyse de la corrélation perplexité <-> justesse
"""

import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_JSONL = Path(__file__).resolve().parent / "triviaqa_results.jsonl"
FIGS_DIR = Path(__file__).resolve().parent / "figures"


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                continue
            rows.append({
                "question_id": obj["question_id"],
                "question": obj["question"],
                "full_answer": obj["full_answer"],
                "ground_truths": obj["ground_truths"],
                "answer_correct": obj["answer_correct"],
                "sentence_perplexity": obj["sentence_perplexity"],
                "avg_log_prob": obj["avg_log_prob"],
                # Perplexity method
                "perp_top_tokens": obj["perplexity_importance"]["top_tokens"],
                "perp_answer_in_top_k": obj["perplexity_importance"]["comparison"]["answer_in_top_k"],
                "perp_answer_rank": obj["perplexity_importance"]["comparison"]["answer_rank"],
                "perp_top_tokens_text": obj["perplexity_importance"]["comparison"]["top_tokens_text"],
                # Cosine method
                "cos_top_tokens": obj["cosine_importance"]["top_tokens"],
                "cos_answer_in_top_k": obj["cosine_importance"]["comparison"]["answer_in_top_k"],
                "cos_answer_rank": obj["cosine_importance"]["comparison"]["answer_rank"],
                "cos_top_tokens_text": obj["cosine_importance"]["comparison"]["top_tokens_text"],
                # Raw token data
                "token_data": obj.get("token_data", []),
            })
    return pd.DataFrame(rows)


def plot_perplexity_vs_correctness(df: pd.DataFrame, out_dir: Path):
    """Boxplot: perplexity distribution for correct vs incorrect answers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    correct = df[df["answer_correct"]]["sentence_perplexity"]
    incorrect = df[~df["answer_correct"]]["sentence_perplexity"]

    if correct.empty or incorrect.empty:
        plt.close(fig)
        return

    bp = ax.boxplot(
        [correct, incorrect],
        labels=["Correct", "Incorrect"],
        patch_artist=True,
        showmeans=True,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    ax.set_ylabel("Perplexité de la phrase")
    ax.set_title("Perplexité de la réponse vs justesse (TriviaQA)")
    fig.tight_layout()
    fig.savefig(out_dir / "perplexity_vs_correctness.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: perplexity_vs_correctness.png")


def plot_method_comparison_hit_rate(df: pd.DataFrame, out_dir: Path):
    """Bar chart: fraction of questions where answer is in top-k important tokens."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["Perplexity", "Cosine Sim."]
    rates = [df["perp_answer_in_top_k"].mean(), df["cos_answer_in_top_k"].mean()]

    bars = ax.bar(methods, rates, color=["#2196F3", "#FF9800"], alpha=0.85)
    ax.set_ylabel("Réponse dans le top-k (%)")
    ax.set_title("Comparaison des méthodes – La réponse est-elle dans les top-k tokens ?")
    ax.set_ylim(0, 1.0)

    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r:.1%}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_dir / "method_comparison_hit_rate.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: method_comparison_hit_rate.png")


def plot_rank_distribution(df: pd.DataFrame, out_dir: Path):
    """Histogram of the rank at which the answer token appears (when found)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, col, label, color in [
        (axes[0], "perp_answer_rank", "Perplexity", "#2196F3"),
        (axes[1], "cos_answer_rank", "Cosine Sim.", "#FF9800"),
    ]:
        ranks = df[col].dropna().astype(int)
        if ranks.empty:
            ax.set_title(f"{label} – aucun hit")
            continue
        max_rank = int(ranks.max())
        ax.hist(ranks, bins=range(1, max_rank + 2), color=color, alpha=0.85,
                edgecolor="white", align="left")
        ax.set_xlabel("Rang du token-réponse")
        ax.set_ylabel("Nombre de questions")
        ax.set_title(f"{label} – Distribution des rangs")
        ax.set_xticks(range(1, max_rank + 1))

    fig.tight_layout()
    fig.savefig(out_dir / "rank_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: rank_distribution.png")


def plot_token_logprob_heatmap(df: pd.DataFrame, out_dir: Path, n_examples: int = 5):
    """
    For a few example questions, show the log-prob of each token as a heatmap bar.
    Highlights which tokens have low probability (high uncertainty).
    """
    sample = df.head(n_examples)
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 2.5 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sample.iterrows()):
        tokens = row["token_data"]
        if not tokens:
            continue
        labels = [t["token"] for t in tokens]
        logps = [t["log_prob"] for t in tokens]

        norm_vals = np.array(logps)
        vmin, vmax = norm_vals.min(), min(norm_vals.max(), 0.0)
        if vmax == vmin:
            colors_val = np.ones_like(norm_vals) * 0.5
        else:
            colors_val = (norm_vals - vmin) / (vmax - vmin)

        cmap = plt.cm.RdYlGn
        colors = [cmap(v) for v in colors_val]

        ax.barh(range(len(labels)), logps, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
        ax.invert_yaxis()
        ax.set_xlabel("log-prob")
        q_short = row["question"][:60] + "..." if len(row["question"]) > 60 else row["question"]
        ax.set_title(f"Q: {q_short}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "token_logprob_examples.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: token_logprob_examples.png")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_JSONL
    if not path.exists():
        print(f"Results file not found: {path}")
        print("Run main_triviaqa.py first.")
        return

    df = load_results(path)
    print(f"Loaded {len(df)} valid results.\n")

    n = len(df)

    # ---- Summary ----
    print("=" * 70)
    print(" RÉSUMÉ GLOBAL")
    print("=" * 70)
    correct_count = df["answer_correct"].sum()
    print(f"  Questions:                        {n}")
    print(f"  Réponse correcte:                 {correct_count}/{n} ({correct_count/n:.1%})")
    print(f"  Perplexité moyenne:               {df['sentence_perplexity'].mean():.2f}")

    perp_hit = df["perp_answer_in_top_k"].sum()
    cos_hit = df["cos_answer_in_top_k"].sum()
    perp_ranks = df["perp_answer_rank"].dropna()
    cos_ranks = df["cos_answer_rank"].dropna()

    print(f"\n  Perplexity method:")
    print(f"    Réponse dans top-k:             {perp_hit}/{n} ({perp_hit/n:.1%})")
    if not perp_ranks.empty:
        print(f"    Rang moyen (quand trouvé):      {perp_ranks.mean():.1f}")

    print(f"\n  Cosine similarity method:")
    print(f"    Réponse dans top-k:             {cos_hit}/{n} ({cos_hit/n:.1%})")
    if not cos_ranks.empty:
        print(f"    Rang moyen (quand trouvé):      {cos_ranks.mean():.1f}")

    # ---- Per-question detail (first 10) ----
    print("\n" + "=" * 70)
    print(" EXEMPLES DÉTAILLÉS (10 premiers)")
    print("=" * 70)
    for _, row in df.head(10).iterrows():
        print(f"\nQ: {row['question']}")
        print(f"  Réponse LLM:       {row['full_answer']}")
        print(f"  Vérité:            {row['ground_truths'][:3]}")
        print(f"  Correct?           {'Oui' if row['answer_correct'] else 'Non'}")
        print(f"  Perplexité:        {row['sentence_perplexity']:.2f}")
        print(f"  Top perp tokens:   {row['perp_top_tokens_text']}")
        print(f"    -> réponse rang: {row['perp_answer_rank']}")
        print(f"  Top cos tokens:    {row['cos_top_tokens_text']}")
        print(f"    -> réponse rang: {row['cos_answer_rank']}")

    # ---- Figures ----
    FIGS_DIR.mkdir(exist_ok=True)
    print(f"\nGenerating figures in {FIGS_DIR}...")
    plot_perplexity_vs_correctness(df, FIGS_DIR)
    plot_method_comparison_hit_rate(df, FIGS_DIR)
    plot_rank_distribution(df, FIGS_DIR)
    plot_token_logprob_heatmap(df, FIGS_DIR, n_examples=5)

    print("\nDone!")


if __name__ == "__main__":
    main()
