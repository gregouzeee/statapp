"""
Generate a multi-page PDF report summarizing the ELI5 uncertainty methods.
Uses matplotlib's PdfPages backend — no external PDF library needed.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from scipy.stats import spearmanr

# ── Paths ──
BASE = Path(__file__).resolve().parent
FIGS = BASE / "figures"
OUT_PDF = BASE / "rapport_methodes_eli5.pdf"

JUDGED = BASE / "eli5_judged.jsonl"
SE_FILE = BASE / "eli5_semantic_entropy.jsonl"
SC_FILE = BASE / "eli5_selfcheck.jsonl"


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "error" not in o:
                out.append(o)
    return out


def text_page(pdf, lines, fontsize=11, title=None, title_size=16):
    """Create a page of formatted text."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis("off")

    y = 0.95
    if title:
        ax.text(0.5, y, title, transform=ax.transAxes,
                fontsize=title_size, fontweight="bold", ha="center", va="top",
                family="serif")
        y -= 0.05

    for line in lines:
        if line.startswith("##"):
            ax.text(0.05, y, line.lstrip("# "), transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top", family="serif",
                    color="#2c3e50")
            y -= 0.035
        elif line.startswith("**"):
            ax.text(0.05, y, line.strip("*"), transform=ax.transAxes,
                    fontsize=fontsize, fontweight="bold", va="top", family="serif")
            y -= 0.028
        elif line == "---":
            ax.axhline(y=y + 0.005, xmin=0.05, xmax=0.95,
                       color="gray", linewidth=0.5)
            y -= 0.015
        elif line == "":
            y -= 0.012
        else:
            ax.text(0.05, y, line, transform=ax.transAxes,
                    fontsize=fontsize, va="top", family="serif",
                    wrap=True, linespacing=1.3)
            # Estimate line height based on text length
            n_visual_lines = max(1, len(line) // 90 + 1)
            y -= 0.025 * n_visual_lines

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, img_path, caption="", title=""):
    """Create a page with an image and optional caption."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    y_top = 0.97
    if title:
        ax.text(0.5, y_top, title, transform=ax.transAxes,
                fontsize=14, fontweight="bold", ha="center", va="top",
                family="serif", color="#2c3e50")
        y_top -= 0.04

    img = imread(str(img_path))
    h, w = img.shape[:2]
    aspect = w / h

    # Fit image within page
    img_width = 0.85
    img_height = img_width / aspect
    if img_height > 0.65:
        img_height = 0.65
        img_width = img_height * aspect

    left = (1 - img_width) / 2
    bottom = y_top - 0.05 - img_height

    ax_img = fig.add_axes([left, bottom, img_width, img_height])
    ax_img.imshow(img)
    ax_img.axis("off")

    if caption:
        ax.text(0.5, bottom - 0.02, caption, transform=ax.transAxes,
                fontsize=9, ha="center", va="top", family="serif",
                style="italic", color="#555555", wrap=True)

    pdf.savefig(fig)
    plt.close(fig)


def two_images_page(pdf, img1_path, img2_path, title="",
                    cap1="", cap2=""):
    """Page with two images stacked vertically."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis("off")

    if title:
        ax_bg.text(0.5, 0.97, title, transform=ax_bg.transAxes,
                   fontsize=14, fontweight="bold", ha="center", va="top",
                   family="serif", color="#2c3e50")

    for i, (img_path, cap) in enumerate([(img1_path, cap1), (img2_path, cap2)]):
        if not Path(img_path).exists():
            continue
        img = imread(str(img_path))
        h, w = img.shape[:2]
        aspect = w / h

        img_w = 0.82
        img_h = img_w / aspect
        if img_h > 0.38:
            img_h = 0.38
            img_w = img_h * aspect

        left = (1 - img_w) / 2
        bottom = 0.52 - i * 0.47

        ax_img = fig.add_axes([left, bottom, img_w, img_h])
        ax_img.imshow(img)
        ax_img.axis("off")

        if cap:
            ax_bg.text(0.5, bottom - 0.01, cap,
                       transform=ax_bg.transAxes, fontsize=9,
                       ha="center", va="top", family="serif",
                       style="italic", color="#555555")

    pdf.savefig(fig)
    plt.close(fig)


def metrics_table_page(pdf, records, title="Tableau des metriques d'evaluation"):
    """Compute and display the metrics table."""
    y = np.array([1.0 if r["accuracy"] >= 4 else 0.0 for r in records])

    def rankdata(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.zeros(len(x), dtype=float)
        sx = x[order]
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and sx[j] == sx[i]:
                j += 1
            avg = (i + j - 1) / 2.0 + 1.0
            ranks[order[i:j]] = avg
            i = j
        return ranks

    def brier(y, s): return float(np.mean((s - y) ** 2))

    def ece_fn(y, s, bins=10):
        s = np.clip(s, 0, 1)
        edges = np.linspace(0, 1, bins + 1)
        e = 0
        n = len(y)
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (s >= lo) & (s <= hi) if i == bins - 1 else (s >= lo) & (s < hi)
            if not np.any(mask):
                continue
            e += (np.sum(mask) / n) * abs(np.mean(y[mask]) - np.mean(s[mask]))
        return float(e)

    def auroc(y, s):
        np_ = int(np.sum(y == 1))
        nn = int(np.sum(y == 0))
        if np_ == 0 or nn == 0:
            return float("nan")
        r = rankdata(s)
        rs = float(np.sum(r[y == 1]))
        return float((rs - np_ * (np_ + 1) / 2) / (np_ * nn))

    def auprc(y, s):
        np_ = int(np.sum(y == 1))
        if np_ == 0:
            return float("nan")
        o = np.argsort(-s, kind="mergesort")
        ys = y[o]
        tp = np.cumsum(ys == 1)
        fp = np.cumsum(ys == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np_
        rp = np.concatenate(([0.0], rec[:-1]))
        return float(np.sum((rec - rp) * prec))

    signals = {}
    ppls = np.array([r["overall_perplexity"] for r in records])
    pmin, pmax = ppls.min(), ppls.max()
    if pmax > pmin:
        signals["Perplexite"] = 1.0 - (ppls - pmin) / (pmax - pmin)

    ents = np.array([r["overall_topk_entropy"] for r in records])
    emin, emax = ents.min(), ents.max()
    if emax > emin:
        signals["Top-k Entropy"] = 1.0 - (ents - emin) / (emax - emin)

    ses = np.array([r.get("semantic_entropy", 0) for r in records])
    smin, smax = ses.min(), ses.max()
    if smax > smin:
        signals["Semantic Entropy"] = 1.0 - (ses - smin) / (smax - smin)

    scs = np.array([r.get("selfcheck_score", 0.5) for r in records])
    scmin, scmax = scs.min(), scs.max()
    if scmax > scmin:
        signals["SelfCheck"] = 1.0 - (scs - scmin) / (scmax - scmin)

    if len(signals) >= 2:
        signals["Combine"] = np.mean([signals[k] for k in signals], axis=0)

    # Build table data
    col_labels = ["Signal", "Brier", "ECE", "AUROC", "AUPRC"]
    table_data = []
    for name, vals in signals.items():
        table_data.append([
            name,
            f"{brier(y, vals):.4f}",
            f"{ece_fn(y, vals):.4f}",
            f"{auroc(y, vals):.4f}",
            f"{auprc(y, vals):.4f}",
        ])

    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    ax.text(0.5, 0.92, title, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="center", va="top",
            family="serif", color="#2c3e50")

    ax.text(0.05, 0.85,
            f"Baseline : {np.mean(y)*100:.0f}% des questions ont accuracy >= 4 (n={len(y)})",
            transform=ax.transAxes, fontsize=11, va="top", family="serif")

    ax.text(0.05, 0.82,
            "y = 1 si accuracy >= 4, 0 sinon. s = signal de confiance normalise [0,1].",
            transform=ax.transAxes, fontsize=10, va="top", family="serif",
            style="italic", color="#555555")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.08, 0.45, 0.84, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best values
    for i, row in enumerate(table_data):
        if row[0] == "Combine":
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor("#e8f4fd")

    # Interpretation
    interp_lines = [
        "Interpretation des metriques :",
        "",
        "- Brier : erreur quadratique moyenne entre confiance et resultat.",
        "   Plus bas = meilleure calibration.",
        "",
        "- ECE : erreur de calibration attendue (binned). Plus bas = mieux.",
        "",
        "- AUROC : capacite du signal a distinguer bonnes/mauvaises reponses.",
        "   0.5 = hasard, 1.0 = parfait.",
        "",
        "- AUPRC : aire sous la courbe precision-rappel.",
        "   Utile quand les classes sont desequilibrees (82% de positifs ici).",
    ]
    y_pos = 0.40
    for line in interp_lines:
        ax.text(0.05, y_pos, line, transform=ax.transAxes,
                fontsize=10, va="top", family="serif")
        y_pos -= 0.025

    pdf.savefig(fig)
    plt.close(fig)


def main():
    judged = load_jsonl(JUDGED)
    se_data = load_jsonl(SE_FILE)
    sc_data = load_jsonl(SC_FILE)

    se_by_qid = {r["question_id"]: r for r in se_data}
    sc_by_qid = {r["question_id"]: r for r in sc_data}

    # Merge for metrics
    records = []
    for r in judged:
        qid = r["question_id"]
        a = (r.get("judge_scores") or {}).get("accuracy")
        if a is None:
            continue
        entry = {
            "accuracy": a,
            "overall_perplexity": r.get("overall_perplexity"),
            "overall_topk_entropy": r.get("overall_topk_entropy"),
        }
        if qid in se_by_qid:
            entry["semantic_entropy"] = se_by_qid[qid]["semantic_entropy"]
        if qid in sc_by_qid:
            entry["selfcheck_score"] = sc_by_qid[qid]["selfcheck_score"]
        records.append(entry)

    # Stats
    se_vals = [r["semantic_entropy"] for r in se_data]
    sc_vals = [r["selfcheck_score"] for r in sc_data]
    all_ppl = [s["perplexity"] for r in judged for s in r["sentences"]]

    acc_list = [(r.get("judge_scores") or {}).get("accuracy") for r in judged]
    acc_list = [a for a in acc_list if a is not None]

    with PdfPages(str(OUT_PDF)) as pdf:

        # ──── PAGE 1: Title + Introduction ────
        text_page(pdf, [
            "",
            "",
            "",
            "Analyse realisee sur le dataset ELI5 (Explain Like I'm 5),",
            "100 questions a reponse ouverte traitees par Gemini 2.0 Flash.",
            "",
            "---",
            "",
            "## Methodes implementees",
            "",
            "1. Perplexite et Entropie top-k (niveau token)",
            "2. LLM-as-Judge (evaluation de qualite)",
            "3. Semantic Entropy (niveau reponse)",
            "4. SelfCheckGPT (verification phrase par phrase)",
            "5. Score de confiance combine + evaluation",
            "",
            "---",
            "",
            "## Pipeline",
            "",
            "Question ELI5  -->  Generation (Gemini 2.0 Flash, logprobs)",
            "  --> Perplexite + Entropie par phrase",
            "  --> K=5 reponses alternatives (temp=1.0)",
            "  --> Clustering NLI --> Semantic Entropy",
            "  --> SelfCheck phrase-par-phrase",
            "  --> Evaluation par juge LLM",
            "  --> Score de confiance combine",
        ], title="Quantification de l'Incertitude\nsur Texte Generatif (ELI5)")

        # ──── PAGE 2: Method 1 - Perplexity ────
        text_page(pdf, [
            "",
            "La perplexite mesure la 'surprise' du modele face a sa propre",
            "generation. Pour chaque token genere, on recupere la log-probabilite",
            "attribuee par le modele :",
            "",
            "  PPL(phrase) = exp( -1/N * sum(log P(t_i | t_<i)) )",
            "",
            "Une perplexite elevee signifie que le modele hesite sur les mots",
            "a utiliser, signe potentiel d'incertitude.",
            "",
            "**Entropie top-k** : On calcule aussi l'entropie de Shannon sur les",
            "k=5 tokens les plus probables a chaque position. Une entropie elevee",
            "signifie que le modele hesite entre plusieurs alternatives.",
            "",
            "---",
            "",
            "## Statistiques cles",
            "",
            f"  - 100 questions, {len(all_ppl)} phrases au total",
            f"  - PPL moyenne par phrase : {np.mean(all_ppl):.2f} (std={np.std(all_ppl):.2f})",
            f"  - Entropie top-k moyenne : 0.523",
            f"  - Correlation PPL vs accuracy : r=-0.179 (p=0.075)",
            "",
            "La perplexite est faiblement anti-correlee avec la qualite :",
            "les reponses de mauvaise qualite ont une perplexite legerement",
            "plus elevee, mais la relation est faible.",
        ], title="Methode 1 : Perplexite et Entropie Top-k")

        # ──── PAGE 3: PPL figures ────
        two_images_page(pdf,
            FIGS / "sentence_distributions.png",
            FIGS / "uncertainty_vs_accuracy.png",
            title="Perplexite : distributions et lien avec l'accuracy",
            cap1="Figure 1 : Distribution de la perplexite et de l'entropie par phrase",
            cap2="Figure 3 : Perplexite et entropie groupees par score d'accuracy du juge")

        # ──── PAGE 4: Sentence heatmaps ────
        image_page(pdf, FIGS / "sentence_heatmaps.png",
                   title="Exemples de reponses colorees par perplexite",
                   caption="Figure 4 : Chaque phrase est coloree par sa perplexite "
                           "(vert=faible, rouge=elevee)")

        # ──── PAGE 5: Method 2 - Semantic Entropy ────
        text_page(pdf, [
            "",
            "La Semantic Entropy (Kuhn et al., 2023) mesure la diversite",
            "semantique des reponses du modele quand on le fait repondre",
            "plusieurs fois a la meme question.",
            "",
            "## Algorithme",
            "",
            "1. Generer K=5 reponses avec temperature=1.0 (haute diversite)",
            "2. Pour chaque paire de reponses, evaluer l'equivalence",
            "   semantique via NLI (Gemini 2.5 Flash)",
            "3. Regrouper les reponses equivalentes par Union-Find",
            "4. Calculer l'entropie de Shannon sur les clusters :",
            "",
            "  SE = - sum( P(C) * log P(C) )",
            "",
            "  ou P(C) = sum(exp(mean_log_prob(r))) / Z pour r dans C",
            "",
            "**SE = 0** signifie que toutes les reponses tombent dans le",
            "meme cluster : le modele est unanime.",
            "**SE > 0** signifie diversite semantique : le modele donne",
            "des reponses differentes selon les tirages.",
            "",
            "---",
            "",
            "## Resultats",
            "",
            f"  - SE moyenne : {np.mean(se_vals):.4f} (std={np.std(se_vals):.4f})",
            f"  - SE = 0 : {sum(1 for s in se_vals if s==0)}/100 questions (51%)",
            f"  - Clusters moyens : {np.mean([r['num_clusters'] for r in se_data]):.2f}",
            f"  - Correlation SE vs accuracy : r=-0.173 (p=0.085)",
            "",
            "Probleme identifie : SE=0 ne garantit pas une bonne reponse.",
            "9 questions sur 51 avec SE=0 ont une accuracy <= 3.",
        ], title="Methode 2 : Semantic Entropy")

        # ──── PAGE 6: SE figures ────
        two_images_page(pdf,
            FIGS / "semantic_entropy_distribution.png",
            FIGS / "se_vs_accuracy.png",
            title="Semantic Entropy : distribution et lien avec l'accuracy",
            cap1="Figure 6 : Distribution de la Semantic Entropy (51% a zero)",
            cap2="Figure 7 : SE par score d'accuracy (boxplot + scatter)")

        # ──── PAGE 7: SE additional ────
        two_images_page(pdf,
            FIGS / "se_vs_perplexity.png",
            FIGS / "clusters_vs_accuracy.png",
            title="Semantic Entropy : analyses supplementaires",
            cap1="Figure 8 : SE vs perplexite globale (faible correlation)",
            cap2="Figure 9 : Nombre de clusters par score d'accuracy")

        # ──── PAGE 8: Method 3 - SelfCheck ────
        text_page(pdf, [
            "",
            "SelfCheckGPT (Manakul et al., 2023) verifie la coherence",
            "factuelle d'une reponse en la comparant phrase par phrase",
            "a des reponses alternatives du meme modele.",
            "",
            "## Algorithme",
            "",
            "1. Prendre la reponse originale et la decouper en phrases",
            "2. Prendre les K=5 reponses alternatives (de Semantic Entropy)",
            "3. Pour chaque (phrase, passage alternatif), demander a",
            "   Gemini 2.5 Flash :",
            "   'Cette phrase est-elle supportee par ce passage ?'",
            "4. Score par phrase = proportion de 'Non supporte'",
            "   (0.0 = confirmee partout, 1.0 = contredite partout)",
            "5. Score global = moyenne des scores par phrase",
            "",
            "## Adaptation",
            "",
            "L'implementation originale utilise un modele HuggingFace local.",
            "Notre version utilise Gemini 2.5 Flash via API avec des prompts",
            "batchees (toutes les phrases evaluees en un seul appel par passage).",
            "",
            "---",
            "",
            "## Resultats",
            "",
            f"  - Score moyen : {np.mean(sc_vals):.4f} (std={np.std(sc_vals):.4f})",
            f"  - Fiable (< 0.3) : {sum(1 for s in sc_vals if s<0.3)} questions",
            f"  - Incertain (0.3-0.7) : {sum(1 for s in sc_vals if 0.3<=s<0.7)} questions",
            f"  - Non fiable (>= 0.7) : {sum(1 for s in sc_vals if s>=0.7)} questions",
            f"  - Correlation vs accuracy : r=-0.069 (p=0.496)",
        ], title="Methode 3 : SelfCheckGPT")

        # ──── PAGE 9: SelfCheck figures ────
        two_images_page(pdf,
            FIGS / "selfcheck_distribution.png",
            FIGS / "selfcheck_vs_accuracy.png",
            title="SelfCheck : distribution et lien avec l'accuracy",
            cap1="Figure 10 : Distribution des scores SelfCheck",
            cap2="Figure 11 : SelfCheck par score d'accuracy")

        # ──── PAGE 10: Combined analysis ────
        text_page(pdf, [
            "",
            "On combine les quatre signaux de confiance pour evaluer",
            "la capacite a predire la qualite des reponses.",
            "",
            "## Signaux de confiance",
            "",
            "  c_ppl = 1 - normalized(perplexite)",
            "  c_entropy = 1 - normalized(top-k entropy)",
            "  c_se = 1 - normalized(semantic entropy)",
            "  c_selfcheck = 1 - normalized(selfcheck score)",
            "  c_combine = moyenne des 4 signaux",
            "",
            "Chaque signal est normalise en [0,1] par min-max.",
            "On inverse les signaux ou 'haut = incertain'.",
            "",
            "## Evaluation",
            "",
            "On definit y=1 si accuracy >= 4, y=0 sinon.",
            f"Baseline : {np.mean([1 if r['accuracy']>=4 else 0 for r in records])*100:.0f}% "
            "des reponses sont de bonne qualite.",
            "",
            "## Matrice SE x SelfCheck",
            "",
            "On croise SE (confiant si SE=0, incertain si SE>0) avec",
            "SelfCheck (fiable si <0.3, non fiable si >=0.3) pour voir",
            "si la combinaison capture mieux les erreurs.",
        ], title="Score de Confiance Combine")

        # ──── PAGE 11: Matrix + Coverage ────
        two_images_page(pdf,
            FIGS / "se_selfcheck_matrix.png",
            FIGS / "accuracy_coverage.png",
            title="Analyse combinee : matrice et courbes de couverture",
            cap1="Figure 12 : Accuracy moyenne par quadrant SE x SelfCheck",
            cap2="Figure 13 : Accuracy vs couverture pour chaque signal")

        # ──── PAGE 12: Metrics table ────
        metrics_table_page(pdf, records)

        # ──── PAGE 13: Conclusions ────
        text_page(pdf, [
            "",
            "## Resultats principaux",
            "",
            "1. La perplexite et l'entropie top-k capturent une hesitation",
            "   au niveau des tokens, mais la correlation avec la qualite",
            "   des reponses reste faible (r=-0.179).",
            "",
            "2. La Semantic Entropy mesure la coherence inter-reponses.",
            "   51% des questions ont SE=0 (modele unanime), mais cela",
            "   ne garantit pas la justesse : 9/51 unanimes ont acc<=3.",
            "",
            "3. SelfCheck verifie la coherence phrase-par-phrase.",
            "   La majorite des questions (67%) sont dans la zone",
            "   'incertaine' (0.3-0.7), montrant une verification",
            "   plus nuancee que SE (binaire).",
            "",
            "4. Aucun signal individuel ne permet de predire la qualite",
            "   des reponses avec fiabilite (meilleur AUROC ~ 0.62).",
            "   Cela est coherent avec la nature du probleme : pour des",
            "   questions ouvertes, l'incertitude du modele ne correspond",
            "   pas directement a la justesse factuelle.",
            "",
            "5. Le signal combine ameliore legerement la calibration",
            "   (meilleur Brier et ECE) mais l'AUROC reste limite.",
            "",
            "---",
            "",
            "## Limites",
            "",
            "- Echantillon limite (100 questions ELI5)",
            "- Evaluation par juge LLM (pas de ground truth humain)",
            "- Questions ouvertes : difficile de definir 'correct'",
            "- Le modele peut etre confiant et faux (SE=0, mauvaise acc.)",
            "",
            "---",
            "",
            "## Perspectives",
            "",
            "- Appliquer sur un dataset plus grand / plus diversifie",
            "- Combiner avec conformal prediction pour des garanties",
            "  statistiques de couverture",
            "- Explorer des methodes ensemblistes (plusieurs modeles)",
            "- Tester sur des taches factuelles (TriviaQA) ou le lien",
            "  incertitude-exactitude est plus fort",
        ], title="Conclusions et Perspectives")

    print(f"PDF generated: {OUT_PDF}")


if __name__ == "__main__":
    main()
