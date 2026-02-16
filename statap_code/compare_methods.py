"""
Comparaison des méthodes d'incertitude sur MMLU (Gemini 2.0 Flash)
==================================================================
Toutes les méthodes sont évaluées sur les MÊMES 107 questions
(celles avec logits A/B/C/D complets).

Méthodes comparées :
  1. Logprobs — signaux c_ent, c_pmax
  2. Confiance déclarée — signaux c_decl, c_ent, c_pmax
  3. Prédiction conforme (LAC & APS) — ensembles avec garantie de couverture

Sorties : figures PNG + rapport PDF récapitulatif.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ENTROPY_RESULTS = ROOT / "entropy_uncertainty" / "results"
CONFORMAL_DIR = ROOT / "Conformal_prediction"
OUT_DIR = ROOT / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)

PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Les 107 UIDs de référence ─────────────────────────────────────────────────
PAIRED_CSV = ENTROPY_RESULTS / "mmlu_conf_compare_paired.csv"
PAIRED_UIDS = set(pd.read_csv(PAIRED_CSV)["uid"].tolist())
N_PAIRED = len(PAIRED_UIDS)
print(f"UIDs de référence : {N_PAIRED} questions")

# ── Style matplotlib ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
})


# =============================================================================
# Utilitaires
# =============================================================================
def load_jsonl_filtered(path, uids):
    """Charge un JSONL et ne garde que les UIDs demandés."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("uid") in uids:
                items.append(rec)
    return items


def accuracy_coverage_curve(y, scores):
    """Courbe accuracy vs coverage (tri par score décroissant)."""
    order = np.argsort(-scores)
    y_sorted = y[order]
    s_sorted = scores[order]
    n = len(y)
    rows = []
    for cov in np.arange(0.1, 1.01, 0.1):
        k = max(1, int(np.ceil(cov * n)))
        acc = float(np.mean(y_sorted[:k]))
        thr = float(s_sorted[k - 1])
        rows.append({"coverage": round(cov, 1), "k": k, "accuracy": acc, "threshold": thr})
    return rows


# =============================================================================
# 1. Courbes Accuracy vs Coverage — Logprobs vs Déclarée (107 questions)
# =============================================================================
def plot_accuracy_coverage():
    """
    Courbes accuracy–coverage sur les 107 questions appariées,
    tous les signaux sur un même graphe.
    """
    df = pd.read_csv(ENTROPY_RESULTS / "mmlu_conf_compare_paired_coverage.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    style = {
        "c_ent_logprob":  ("Entropie (logprobs)",     "#2196F3", "-",  "o"),
        "c_pmax_logprob": ("Prob. max (logprobs)",     "#4CAF50", "-",  "s"),
        "c_ent_decl":     ("Entropie (déclarée)",      "#2196F3", "--", "^"),
        "c_pmax_decl":    ("Prob. max (déclarée)",     "#4CAF50", "--", "D"),
        "c_decl":         ("Confiance déclarée brute", "#E91E63", "--", "x"),
    }

    for sig, (label, color, ls, marker) in style.items():
        data = df[df["signal"] == sig]
        if data.empty:
            continue
        ax.plot(
            data["coverage"], data["accuracy"],
            marker=marker, markersize=5, linewidth=2,
            color=color, linestyle=ls, label=label,
        )

    ax.set_xlabel("Couverture (fraction retenue)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Couverture — Logprobs vs Déclarée (n={N_PAIRED})")
    ax.set_xlim(0.05, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig1_accuracy_coverage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")
    return path


# =============================================================================
# 2. Prédiction conforme (LAC / APS) sur les 107 questions
# =============================================================================
def run_conformal_on_paired():
    """
    LAC et APS sur les 107 questions (fichier logit temp0_5 filtré).
    """
    from statap_code.Conformal_prediction.conformal_prediction import (
        load_jsonl, evaluate, probs_vector, true_label,
        lac_score, aps_score, conformal_quantile,
        lac_predict_set, aps_predict_set,
    )

    # Charger et filtrer sur les 107 UIDs
    all_items = load_jsonl(str(CONFORMAL_DIR / "logit_mmlu_500_temp0_5.jsonl"))
    items = [it for it in all_items if it.get("uid") in PAIRED_UIDS]
    print(f"  Conformal prediction sur {len(items)} questions")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = []

    for alpha in alphas:
        for method in ("lac", "aps"):
            res = evaluate(items, alpha=alpha, method=method, seed=0, cal_frac=0.5)
            results.append(res)

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "conformal_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    # ── Graphe LAC vs APS ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method, ls, color in [("lac", "-", "#1565C0"), ("aps", "--", "#F57C00")]:
        sub = df[df["method"] == method]
        axes[0].plot(sub["alpha"], sub["coverage"], marker="o", linestyle=ls,
                     color=color, label=f"{method.upper()}", linewidth=2)
        axes[1].plot(sub["alpha"], sub["avg_set_size"], marker="o", linestyle=ls,
                     color=color, label=f"{method.upper()}", linewidth=2)

    axes[0].plot(alphas, [1 - a for a in alphas], "k--", alpha=0.4, label="Target (1-α)")
    axes[0].set_xlabel("α (taux d'erreur nominal)")
    axes[0].set_ylabel("Couverture empirique")
    axes[0].set_title("Couverture vs α")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("α (taux d'erreur nominal)")
    axes[1].set_ylabel("Taille moyenne de l'ensemble")
    axes[1].set_title("Taille d'ensemble vs α")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Prédiction conforme — LAC vs APS (n={N_PAIRED})", fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig2_conformal_prediction.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    return df


# =============================================================================
# 3. Comparaison unifiée : accuracy–coverage pour les 3 familles de méthodes
# =============================================================================
def plot_unified_comparison():
    """
    Graphe central : superpose les courbes accuracy–coverage des 3 approches
    sur les mêmes 107 questions.
    - Logprobs : c_ent (entropie sur logits)
    - Déclarée : c_decl (confiance verbalisée)
    - Conformal : accuracy du top-1 dans les ensembles LAC à différents α
      → on trace coverage = 1-α vs top1_accuracy
    """
    from statap_code.Conformal_prediction.conformal_prediction import (
        load_jsonl, probs_vector, true_label,
        lac_score, conformal_quantile, lac_predict_set,
    )
    import random

    # ── Logprobs & déclarée (depuis le CSV paired) ──
    df_cov = pd.read_csv(ENTROPY_RESULTS / "mmlu_conf_compare_paired_coverage.csv")

    # ── Conformal : évaluer sur une grille fine d'alpha ──
    all_items = load_jsonl(str(CONFORMAL_DIR / "logit_mmlu_500_temp0_5.jsonl"))
    items = [it for it in all_items if it.get("uid") in PAIRED_UIDS]

    rnd = random.Random(0)
    items_shuffled = items[:]
    rnd.shuffle(items_shuffled)
    n = len(items_shuffled)
    n_cal = n // 2
    cal = items_shuffled[:n_cal]
    test = items_shuffled[n_cal:]

    # Calibration scores LAC
    cal_scores = []
    for it in cal:
        p = probs_vector(it)
        y = true_label(it)
        cal_scores.append(lac_score(p, y))

    # Évaluer pour chaque alpha
    conformal_points = []
    for alpha in np.arange(0.01, 0.51, 0.01):
        qhat = conformal_quantile(cal_scores, alpha)
        cover = 0
        sizes = []
        for it in test:
            p = probs_vector(it)
            y = true_label(it)
            C = lac_predict_set(p, qhat)
            cover += (y in C)
            sizes.append(len(C))
        coverage = cover / len(test)
        avg_size = sum(sizes) / len(sizes)
        conformal_points.append({
            "alpha": alpha,
            "coverage": coverage,
            "avg_set_size": avg_size,
        })

    df_conf = pd.DataFrame(conformal_points)

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(10, 6))

    # Logprobs — entropie
    data = df_cov[df_cov["signal"] == "c_ent_logprob"]
    ax.plot(data["coverage"], data["accuracy"],
            marker="o", markersize=6, linewidth=2.5,
            color="#1565C0", label="Entropie (logprobs)")

    # Logprobs — pmax
    data = df_cov[df_cov["signal"] == "c_pmax_logprob"]
    ax.plot(data["coverage"], data["accuracy"],
            marker="s", markersize=5, linewidth=2,
            color="#2E7D32", label="Prob. max (logprobs)")

    # Déclarée
    data = df_cov[df_cov["signal"] == "c_decl"]
    ax.plot(data["coverage"], data["accuracy"],
            marker="D", markersize=5, linewidth=2,
            color="#E91E63", linestyle="--", label="Confiance déclarée")

    # Conformal LAC — coverage empirique vs accuracy (= coverage car on mesure si y ∈ C)
    ax.plot(df_conf["coverage"], 1 - df_conf["alpha"],
            marker="^", markersize=5, linewidth=2,
            color="#FF6F00", linestyle="-.", label="Conformal LAC (couv. emp. vs target)")

    ax.set_xlabel("Couverture / Fraction retenue")
    ax.set_ylabel("Accuracy / Target")
    ax.set_title(f"Comparaison des 3 approches d'incertitude (n={N_PAIRED})")
    ax.set_xlim(0.05, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig3_unified_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")
    return path


# =============================================================================
# 4. Distribution des scores de confiance (correctes vs incorrectes)
# =============================================================================
def plot_confidence_distributions():
    df = pd.read_csv(PAIRED_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    signals = [
        ("c_ent_logprob",  "y_logprob", "Entropie (logprobs)"),
        ("c_pmax_logprob", "y_logprob", "Prob. max (logprobs)"),
        ("c_ent_decl",     "y_decl",    "Entropie (déclarée)"),
        ("c_pmax_decl",    "y_decl",    "Prob. max (déclarée)"),
    ]

    for ax, (sig, y_col, title) in zip(axes.flat, signals):
        correct = df[df[y_col] == 1][sig].dropna()
        incorrect = df[df[y_col] == 0][sig].dropna()

        bins = np.linspace(0, 1, 21)
        ax.hist(correct, bins=bins, alpha=0.6, color="#4CAF50",
                label=f"Correctes (n={len(correct)})", density=True)
        ax.hist(incorrect, bins=bins, alpha=0.6, color="#F44336",
                label=f"Incorrectes (n={len(incorrect)})", density=True)
        ax.set_xlabel("Score de confiance")
        ax.set_ylabel("Densité")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(f"Distribution des scores — Correctes vs Incorrectes (n={N_PAIRED})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig4_confidence_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")
    return path


# =============================================================================
# 5. Scatter logprobs vs déclarée
# =============================================================================
def plot_logprob_vs_declared():
    df = pd.read_csv(PAIRED_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    correct = df[df["y_logprob"] == 1]
    incorrect = df[df["y_logprob"] == 0]

    for ax, (sig_decl, sig_log, title) in zip(axes, [
        ("c_ent_decl",  "c_ent_logprob",  "Entropie : Logprobs vs Déclarée"),
        ("c_pmax_decl", "c_pmax_logprob", "Prob. max : Logprobs vs Déclarée"),
    ]):
        ax.scatter(correct[sig_decl], correct[sig_log],
                   c="#4CAF50", alpha=0.6, s=30, label="Correcte", zorder=3)
        ax.scatter(incorrect[sig_decl], incorrect[sig_log],
                   c="#F44336", alpha=0.6, s=30, label="Incorrecte", zorder=3)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Confiance (déclarée)")
        ax.set_ylabel("Confiance (logprobs)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f"Confiance Logprobs vs Déclarée — par question (n={N_PAIRED})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig5_logprob_vs_declared.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")
    return path


# =============================================================================
# 6. Tableau récapitulatif
# =============================================================================
def build_summary_table():
    df_cov = pd.read_csv(ENTROPY_RESULTS / "mmlu_conf_compare_paired_coverage.csv")
    rows = []

    signal_info = {
        "c_ent_logprob":  ("Logprobs",  "Entropie"),
        "c_pmax_logprob": ("Logprobs",  "Prob. max"),
        "c_ent_decl":     ("Déclarée",  "Entropie"),
        "c_pmax_decl":    ("Déclarée",  "Prob. max"),
        "c_decl":         ("Déclarée",  "Conf. brute"),
    }

    for sig, (source, signal_name) in signal_info.items():
        data = df_cov[df_cov["signal"] == sig]
        if data.empty:
            continue
        acc_full = data.loc[data["coverage"] == 1.0, "accuracy"].values
        acc_50 = data.loc[data["coverage"] == 0.5, "accuracy"].values
        acc_10 = data.loc[data["coverage"] == 0.1, "accuracy"].values
        rows.append({
            "Source": source,
            "Signal": signal_name,
            "Acc. globale": f"{acc_full[0]:.1%}" if len(acc_full) else "—",
            "Acc. top 50%": f"{acc_50[0]:.1%}" if len(acc_50) else "—",
            "Acc. top 10%": f"{acc_10[0]:.1%}" if len(acc_10) else "—",
        })

    table = pd.DataFrame(rows)
    path = OUT_DIR / "table_summary.csv"
    table.to_csv(path, index=False)
    print(f"  -> {path}")
    return table


# =============================================================================
# 7. Génération du rapport PDF
# =============================================================================
def generate_pdf(figures, table, conformal_df):
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = OUT_DIR / "rapport_comparaison_incertitude.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # ── Page de titre ──
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.65, "Comparaison des méthodes\nd'incertitude", fontsize=24,
                 ha="center", va="center", fontweight="bold")
        fig.text(0.5, 0.50, "MMLU — Gemini 2.0 Flash", fontsize=16,
                 ha="center", va="center", color="gray")
        fig.text(0.5, 0.42,
                 f"Évaluation sur les mêmes {N_PAIRED} questions\n(logits A/B/C/D complets)",
                 fontsize=12, ha="center", va="center")
        fig.text(0.5, 0.32,
                 "Logprobs vs Confiance déclarée vs Prédiction conforme\n"
                 "Signaux : Entropie normalisée, Probabilité max\n"
                 "Méthodes conformes : LAC, APS",
                 fontsize=10, ha="center", va="center", color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # ── Figures ──
        for fig_path in figures:
            if not Path(fig_path).exists():
                continue
            img = plt.imread(str(fig_path))
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0.5)
            pdf.savefig(fig)
            plt.close(fig)

        # ── Tableau récapitulatif ──
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.set_title(f"Tableau récapitulatif — Accuracy par signal (n={N_PAIRED})",
                     fontsize=13, pad=20)
        tbl = ax.table(
            cellText=table.values.tolist(), colLabels=table.columns.tolist(),
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.8)
        for j in range(len(table.columns)):
            tbl[0, j].set_facecolor("#1565C0")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Tableau conformal ──
        if conformal_df is not None and not conformal_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis("off")
            ax.set_title(f"Prédiction conforme — Résultats (n={N_PAIRED})",
                         fontsize=13, pad=20)
            display_df = conformal_df[["method", "alpha", "coverage",
                                        "avg_set_size", "top1_acc", "qhat"]].copy()
            display_df["coverage"] = display_df["coverage"].map(lambda x: f"{x:.3f}")
            display_df["avg_set_size"] = display_df["avg_set_size"].map(lambda x: f"{x:.2f}")
            display_df["top1_acc"] = display_df["top1_acc"].map(lambda x: f"{x:.3f}")
            display_df["qhat"] = display_df["qhat"].map(lambda x: f"{x:.4f}")
            display_df.columns = ["Méthode", "α", "Couverture",
                                   "Taille moy.", "Acc. top-1", "q̂"]
            tbl = ax.table(
                cellText=display_df.values.tolist(),
                colLabels=display_df.columns.tolist(),
                loc="center", cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.1, 1.5)
            for j in range(len(display_df.columns)):
                tbl[0, j].set_facecolor("#1565C0")
                tbl[0, j].set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\n  => PDF final : {pdf_path}")
    return pdf_path


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print(f"Comparaison des méthodes d'incertitude — {N_PAIRED} questions")
    print("=" * 60)

    figures = []

    print("\n[1] Accuracy vs Coverage (logprobs vs déclarée)...")
    figures.append(plot_accuracy_coverage())

    print("\n[2] Prédiction conforme (LAC/APS)...")
    conformal_df = run_conformal_on_paired()
    figures.append(OUT_DIR / "fig2_conformal_prediction.png")

    print("\n[3] Comparaison unifiée des 3 approches...")
    figures.append(plot_unified_comparison())

    print("\n[4] Distribution des scores de confiance...")
    figures.append(plot_confidence_distributions())

    print("\n[5] Scatter logprob vs déclarée...")
    figures.append(plot_logprob_vs_declared())

    print("\n[6] Tableau récapitulatif...")
    table = build_summary_table()
    print(table.to_string(index=False))

    print("\n[7] Génération du PDF...")
    generate_pdf(figures, table, conformal_df)

    print("\n" + "=" * 60)
    print("Terminé ! Tous les fichiers sont dans :")
    print(f"  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
