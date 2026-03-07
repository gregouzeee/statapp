"""
Génère un PDF explicatif des méthodes d'incertitude sur réponses textuelles.
Usage: python -m statap_code.triviaqa_uncertainty.generate_methodes_pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path

OUT_PDF = Path(__file__).resolve().parent / "methodes_incertitude_texte.pdf"


def title_page(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis("off")
    ax.text(0.5, 0.65, "Localisation de l'information\ndans les réponses textuelles de LLM",
            ha="center", va="center", fontsize=22, fontweight="bold",
            linespacing=1.5)
    ax.text(0.5, 0.50, "Deux méthodes basées sur les log-probabilités",
            ha="center", va="center", fontsize=14, color="#555555")
    ax.text(0.5, 0.40, "Perplexity-based token importance\nvs\nCosine similarity leave-one-out",
            ha="center", va="center", fontsize=13, linespacing=1.6, style="italic")
    ax.text(0.5, 0.25, "Dataset : TriviaQA  ·  Modèle : Gemini 2.0 Flash",
            ha="center", va="center", fontsize=11, color="#777777")
    pdf.savefig(fig)
    plt.close(fig)


def page_pipeline(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "1. Pipeline global", ha="center", fontsize=18,
            fontweight="bold")

    steps = [
        ("1", "Charger TriviaQA", "Questions à réponses courtes\n(ex: \"What is the capital of France?\")"),
        ("2", "Prompt → Gemini", "\"Answer in exactly ONE short sentence.\"\n→ Le modèle répond : \"The capital of France is Paris.\""),
        ("3", "Extraire les log-probs", "Gemini renvoie pour chaque token généré :\n· sa log-probabilité\n· les top-k candidats alternatifs avec leurs log-probs"),
        ("4", "Méthode A ou B", "Classer les tokens par importance\n→ identifier ceux qui portent l'information"),
        ("5", "Comparer à la vérité", "La réponse TriviaQA est-elle parmi\nles top-k tokens les plus importants ?"),
    ]

    y = 0.85
    for num, title, desc in steps:
        # Box
        rect = mpatches.FancyBboxPatch((0.08, y - 0.065), 0.84, 0.11,
                                        boxstyle="round,pad=0.01",
                                        facecolor="#E3F2FD", edgecolor="#1976D2",
                                        linewidth=1.5)
        ax.add_patch(rect)
        # Number circle
        circle = plt.Circle((0.14, y), 0.025, color="#1976D2", zorder=5)
        ax.add_patch(circle)
        ax.text(0.14, y, num, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=6)
        # Title
        ax.text(0.20, y + 0.015, title, fontsize=12, fontweight="bold",
                va="center")
        # Description
        ax.text(0.20, y - 0.025, desc, fontsize=9, va="top", color="#333333",
                linespacing=1.4)

        y -= 0.155

    pdf.savefig(fig)
    plt.close(fig)


def page_logprobs_explained(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "2. Qu'est-ce qu'une log-probabilité ?", ha="center",
            fontsize=18, fontweight="bold")

    text = """Quand un LLM génère du texte, il produit les tokens un par un. À chaque position,
il calcule une distribution de probabilité sur tout son vocabulaire.

La log-probabilité d'un token est :  log P(token | contexte précédent)

· log P ≈ 0     →  le modèle est très sûr de ce token  (P ≈ 1)
· log P << 0    →  le modèle hésite, ce token était improbable

Exemple : "The capital of France is Paris."
"""
    ax.text(0.08, 0.87, text, fontsize=10.5, va="top", linespacing=1.6,
            fontfamily="serif")

    # Token log-prob visualization
    tokens =  ["The",   "capital", "of",    "France", "is",    "Paris",  "."]
    logps =   [-0.05,   -0.8,      -0.02,   -1.5,     -0.03,   -4.2,     -0.01]

    bar_y = 0.52
    bar_h = 0.04
    total_w = 0.80
    x_start = 0.10

    # Normalize widths by token count
    w = total_w / len(tokens)

    for i, (tok, lp) in enumerate(zip(tokens, logps)):
        x = x_start + i * w
        # Color: green (high prob) to red (low prob)
        norm_val = max(0, min(1, (lp - (-5)) / (0 - (-5))))
        color = plt.cm.RdYlGn(norm_val)
        rect = mpatches.FancyBboxPatch((x + 0.005, bar_y), w - 0.01, bar_h,
                                        boxstyle="round,pad=0.003",
                                        facecolor=color, edgecolor="#333",
                                        linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, bar_y + bar_h + 0.015, tok, ha="center",
                fontsize=10, fontweight="bold")
        ax.text(x + w / 2, bar_y - 0.015, f"{lp:.2f}", ha="center",
                fontsize=8.5, color="#555")

    ax.text(0.08, bar_y - 0.04, "log P :", fontsize=9, color="#555")

    # Legend
    ax.text(0.08, bar_y - 0.07, "Vert = haute probabilité (token prévisible)      "
            "Rouge = basse probabilité (token informatif / incertain)",
            fontsize=9, color="#555", style="italic")

    # Top-k explanation
    topk_text = """En plus du token choisi, Gemini renvoie les top-k candidats alternatifs :

    Position "Paris" → top-5 candidats :
        Paris    log P = -4.2
        London   log P = -4.5
        Berlin   log P = -5.1
        Rome     log P = -5.8
        Madrid   log P = -6.0

Ces vecteurs de candidats sont utilisés par la méthode cosine similarity."""
    ax.text(0.08, 0.35, topk_text, fontsize=10, va="top", linespacing=1.5,
            fontfamily="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def page_method_perplexity(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "3. Méthode A – Perplexity-based importance",
            ha="center", fontsize=18, fontweight="bold")

    # Intuition
    rect = mpatches.FancyBboxPatch((0.06, 0.83), 0.88, 0.08,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#FFF3E0", edgecolor="#FF9800",
                                    linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.87, "Intuition : un token avec une log-probabilité basse est un token\n"
            "sur lequel le modèle hésite → il porte probablement de l'information.",
            ha="center", va="center", fontsize=10.5, linespacing=1.5)

    text1 = """Principe

Pour chaque token de la réponse du LLM, on définit :

    importance(token) = − log P(token | contexte)

C'est simplement la négation de la log-probabilité. Plus la valeur est grande,
plus le token est "surprenant" pour le modèle → plus il est important.

On filtre les tokens de ponctuation et espaces, puis on trie par importance
décroissante. Les top-k sont nos "tokens informatifs"."""
    ax.text(0.08, 0.80, text1, fontsize=10.5, va="top", linespacing=1.55,
            fontfamily="serif")

    # Worked example
    ax.text(0.08, 0.48, "Exemple concret", fontsize=13, fontweight="bold")

    example = """  Question : "What is the capital of France?"
  Réponse :  "The capital of France is Paris."

  Token        log P     importance = −log P
  ─────        ─────     ───────────────────
  The          −0.05     0.05   (très prévisible, pas important)
  capital      −0.80     0.80
  of           −0.02     0.02   (mot fonctionnel)
  France       −1.50     1.50
  is           −0.03     0.03   (mot fonctionnel)
  Paris        −4.20     4.20   ← LE PLUS IMPORTANT
  .            (filtré)

  Top-3 : Paris (rang 1), France (rang 2), capital (rang 3)
  → La réponse "Paris" est bien au rang 1 ✓"""
    ax.text(0.08, 0.44, example, fontsize=9.5, va="top", linespacing=1.45,
            fontfamily="monospace")

    # Why it works
    ax.text(0.08, 0.13, "Pourquoi ça marche ?", fontsize=13, fontweight="bold")
    why = """Les mots fonctionnels (the, is, of, a) sont très prévisibles → log P ≈ 0.
Les noms propres, dates, chiffres sont rares dans le vocabulaire et dépendent
fortement de la question → log P << 0. L'information se concentre naturellement
dans les tokens à basse probabilité."""
    ax.text(0.08, 0.10, why, fontsize=10.5, va="top", linespacing=1.5,
            fontfamily="serif")

    pdf.savefig(fig)
    plt.close(fig)


def page_method_cosine_1(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "4. Méthode B – Cosine similarity (leave-one-out)",
            ha="center", fontsize=18, fontweight="bold")

    # Intuition box
    rect = mpatches.FancyBboxPatch((0.06, 0.83), 0.88, 0.08,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#E8F5E9", edgecolor="#4CAF50",
                                    linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, 0.87, "Intuition : si retirer un token change beaucoup le \"profil\n"
            "de confiance\" global de la phrase, alors ce token était important.",
            ha="center", va="center", fontsize=10.5, linespacing=1.5)

    text = """Étape 1 — Vocabulaire partagé

À chaque position, Gemini renvoie les top-k candidats (ici k=5). On collecte
tous les tokens qui apparaissent dans n'importe quel top-k → vocabulaire V.

    Position 0 ("The"):     { The: −0.1,  A: −2.0,    This: −3.1  }
    Position 1 ("capital"):  { capital: −0.5,  city: −1.5, largest: −2.8 }
    Position 2 ("is"):      { is: −0.05, was: −3.0,  remains: −4.2 }
    Position 3 ("Paris"):   { Paris: −3.2, London: −3.5, Berlin: −4.0 }

    → V = { The, A, This, capital, city, largest, is, was, remains,
             Paris, London, Berlin }    (|V| = 12)


Étape 2 — Un vecteur par position

Pour chaque position i, on crée un vecteur de dimension |V|.
Si un token t ∈ V apparaît dans le top-k de la position i,
on met sa log-prob ; sinon on met −50 (≈ "probabilité nulle").

    vec₀ = [ −0.1,  −2.0, −3.1,  −50,  −50,  −50, −50, −50,  −50,  −50,   −50,   −50  ]
               The    A    This   cap   city   lrg   is   was   rem  Paris  Lond  Berl

    vec₃ = [ −50,   −50,  −50,   −50,  −50,  −50, −50, −50,  −50,  −3.2,  −3.5,  −4.0 ]

Chaque vecteur capture "à quoi le modèle pensait" à cette position."""
    ax.text(0.08, 0.80, text, fontsize=9.5, va="top", linespacing=1.45,
            fontfamily="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def page_method_cosine_2(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "4. Méthode B – Cosine similarity (suite)",
            ha="center", fontsize=18, fontweight="bold")

    text = """Étape 3 — Vecteur de la phrase complète

On fait la moyenne de tous les vecteurs-position :

    vec_full = mean(vec₀, vec₁, vec₂, vec₃)

C'est le "profil moyen de confiance" de toute la phrase.


Étape 4 — Leave-one-out

Pour chaque token t à la position i, on recalcule la moyenne
sans cette position :

    vec_sans_i = mean(tous les vec SAUF vec_i)

Puis on calcule :

    importance(t) = 1 − cosine_similarity(vec_full, vec_sans_i)

· Si cos ≈ 1 → retirer ce token ne change presque rien → PEU important
                (ex: "The", "is" — profils très communs)

· Si cos << 1 → retirer ce token change beaucoup le profil → TRÈS important
                (ex: "Paris" — profil top-k très différent du reste)"""
    ax.text(0.08, 0.87, text, fontsize=10, va="top", linespacing=1.5,
            fontfamily="monospace")

    # Visual: cosine similarity diagram
    ax.text(0.08, 0.38, "Schéma", fontsize=13, fontweight="bold")

    # Draw two scenarios side by side
    # Left: removing "is" (cos ≈ 1)
    ax.text(0.25, 0.33, 'Retirer "is"', ha="center", fontsize=11, fontweight="bold",
            color="#4CAF50")
    # Arrow from full to without
    ax.annotate("", xy=(0.32, 0.26), xytext=(0.18, 0.26),
                arrowprops=dict(arrowstyle="<->", color="#4CAF50", lw=2))
    ax.text(0.25, 0.275, "cos = 0.998", ha="center", fontsize=10, color="#4CAF50")
    ax.text(0.25, 0.22, "importance = 0.002\n→ pas important", ha="center",
            fontsize=9.5, color="#4CAF50", linespacing=1.4)

    # Right: removing "Paris" (cos drops)
    ax.text(0.70, 0.33, 'Retirer "Paris"', ha="center", fontsize=11, fontweight="bold",
            color="#F44336")
    ax.annotate("", xy=(0.77, 0.26), xytext=(0.63, 0.26),
                arrowprops=dict(arrowstyle="<->", color="#F44336", lw=2))
    ax.text(0.70, 0.275, "cos = 0.85", ha="center", fontsize=10, color="#F44336")
    ax.text(0.70, 0.22, "importance = 0.15\n→ très important !", ha="center",
            fontsize=9.5, color="#F44336", linespacing=1.4)

    # Why it works
    ax.text(0.08, 0.13, "Pourquoi ça marche ?", fontsize=13, fontweight="bold")
    why = """Les tokens informatifs (noms propres, chiffres, dates) ont des top-k candidats
très spécifiques (Paris/London/Berlin) qui ne ressemblent pas au reste de la phrase.
Quand on retire leur vecteur, la moyenne change beaucoup. Les mots fonctionnels
ont des candidats banals (the/a/this) qui ne pèsent pas sur la moyenne."""
    ax.text(0.08, 0.10, why, fontsize=10.5, va="top", linespacing=1.5,
            fontfamily="serif")

    pdf.savefig(fig)
    plt.close(fig)


def page_comparison(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "5. Comparaison avec la vérité TriviaQA",
            ha="center", fontsize=18, fontweight="bold")

    text1 = """Évaluation

Pour chaque question, après avoir classé les tokens par importance :

  1. On prend les top-k tokens les plus importants (ex: k=3)
  2. On regarde si la réponse TriviaQA est dans un de ces tokens
  3. Si oui, on note son rang (1 = le plus important)

C'est la question la plus directe : est-ce que la méthode a identifié
le bon token comme porteur de l'information ?"""
    ax.text(0.08, 0.87, text1, fontsize=10.5, va="top", linespacing=1.55,
            fontfamily="serif")

    # Example table
    ax.text(0.08, 0.62, "Exemple", fontsize=13, fontweight="bold")

    example = """  Question :  "Which planet is known as the Red Planet?"
  Réponse :   "Mars is known as the Red Planet."
  Vérité :    ["Mars"]

  Méthode Perplexity :
    Top-3 : ["Mars", "Red", "Planet"]  →  réponse rang 1 ✓

  Méthode Cosine :
    Top-3 : ["Red", "Planet", "Mars"]  →  réponse rang 3 ✓"""
    ax.text(0.08, 0.58, example, fontsize=10, va="top", linespacing=1.45,
            fontfamily="monospace")

    # Metrics explanation
    ax.text(0.08, 0.36, "Métriques agrégées", fontsize=13, fontweight="bold")

    metrics = """Sur l'ensemble des N questions :

  · Hit rate :  fraction des questions où la réponse est dans le top-k

        hit rate = |{ q : réponse ∈ top-k(q) }| / N

  · Rang moyen :  quand la réponse est trouvée, à quel rang ?
    (1 = la méthode met toujours la réponse en premier, k = en dernier)

        rang moyen = mean(rang(q))  pour q tels que réponse ∈ top-k(q)

  · Corrélation perplexité–justesse :
    Est-ce que les phrases avec une haute perplexité (modèle incertain)
    correspondent à des réponses fausses ? On calcule la corrélation
    point-bisériale entre perplexité (continue) et justesse (binaire)."""
    ax.text(0.08, 0.33, metrics, fontsize=10, va="top", linespacing=1.45,
            fontfamily="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def page_comparison_methods(pdf):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "6. Perplexity vs Cosine — Quand préférer l'une ou l'autre ?",
            ha="center", fontsize=16, fontweight="bold")

    # Table header
    headers = ["", "Perplexity", "Cosine Similarity"]
    col_x = [0.08, 0.30, 0.62]

    y = 0.86
    for x, h in zip(col_x, headers):
        ax.text(x, y, h, fontsize=11, fontweight="bold", va="center")
    ax.plot([0.06, 0.94], [y - 0.015, y - 0.015], color="#999", linewidth=0.8)

    rows = [
        ("Calcul", "−log P(token)", "1 − cos(full, full\\token)"),
        ("Complexité", "O(n)", "O(n × |V|)"),
        ("Données\nnécessaires", "Log-prob du token choisi", "Top-k candidats\navec log-probs"),
        ("Ce que ça\nmesure", "Surprise du modèle\nsur chaque token", "Impact du token sur le\nprofil global de confiance"),
        ("Force", "Simple, direct,\ninterprétable", "Capture les interactions\nentre positions"),
        ("Faiblesse", "Ignore les alternatives\n(ne voit que le token choisi)", "Sensible au choix de k\net au vocabulaire partagé"),
    ]

    y -= 0.04
    for label, perp, cos in rows:
        n_lines = max(label.count("\n"), perp.count("\n"), cos.count("\n")) + 1
        h = 0.03 + n_lines * 0.025
        # Light background for alternating rows
        rect = mpatches.FancyBboxPatch(
            (0.06, y - h + 0.005), 0.88, h,
            boxstyle="round,pad=0.005",
            facecolor="#F5F5F5", edgecolor="none")
        ax.add_patch(rect)

        ax.text(col_x[0], y - h / 2 + 0.01, label, fontsize=9.5, va="center",
                fontweight="bold", linespacing=1.4)
        ax.text(col_x[1], y - h / 2 + 0.01, perp, fontsize=9.5, va="center",
                linespacing=1.4)
        ax.text(col_x[2], y - h / 2 + 0.01, cos, fontsize=9.5, va="center",
                linespacing=1.4)
        y -= h + 0.015

    # Bottom note
    ax.text(0.08, 0.18, "Résumé", fontsize=13, fontweight="bold")
    summary = """La méthode perplexity est plus simple et probablement suffisante pour
les réponses courtes (TriviaQA). La méthode cosine capture des effets plus subtils
en tenant compte de ce que le modèle aurait pu dire à chaque position, mais au prix
d'une plus grande complexité et d'une dépendance au paramètre k.

L'intérêt de comparer les deux est de vérifier si elles identifient les mêmes tokens
comme importants, ou si l'une capture des signaux que l'autre manque."""
    ax.text(0.08, 0.15, summary, fontsize=10.5, va="top", linespacing=1.5,
            fontfamily="serif")

    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(str(OUT_PDF)) as pdf:
        title_page(pdf)
        page_pipeline(pdf)
        page_logprobs_explained(pdf)
        page_method_perplexity(pdf)
        page_method_cosine_1(pdf)
        page_method_cosine_2(pdf)
        page_comparison(pdf)
        page_comparison_methods(pdf)

    print(f"PDF generated: {OUT_PDF}")


if __name__ == "__main__":
    main()
