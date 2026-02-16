"""
Génère un PDF explicatif : chaque figure est accompagnée d'une explication
détaillée (axes, courbes, interprétation).
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "comparison_results"
PDF_PATH = OUT_DIR / "rapport_explique.pdf"

FIGURES = [
    (
        OUT_DIR / "fig1_accuracy_coverage.png",
        "Figure 1 — Accuracy vs Couverture : Logprobs vs Confiance déclarée",
        """\
Protocole. On dispose de 107 questions MMLU pour lesquelles Gemini 2.0 Flash \
a fourni à la fois les logits complets (A/B/C/D) et une confiance verbalisée. \
Pour chaque question, on calcule un score de confiance, puis on trie les \
questions de la plus confiante à la moins confiante.

Axe X — Couverture (fraction retenue). C'est la proportion des 107 questions \
que l'on garde, en partant des plus confiantes. Par exemple, couverture = 0.3 \
signifie qu'on ne garde que les 30% de questions les plus confiantes \
(soit ~33 questions sur 107).

Axe Y — Accuracy. C'est le taux de bonnes réponses parmi les questions \
retenues. C'est un résultat calculé : on regarde, parmi les k questions les \
plus confiantes, combien sont correctement prédites.

Courbes tracées (5 signaux de confiance) :
• Entropie (logprobs) [bleu, trait plein] : confiance = 1 − entropie \
normalisée de la distribution softmax issue des logits.
• Prob. max (logprobs) [vert, trait plein] : confiance = max(P(A), P(B), \
P(C), P(D)) issue des logits.
• Entropie (déclarée) [bleu, pointillé] : même formule d'entropie, mais \
appliquée aux probabilités que le modèle a lui-même verbalisées.
• Prob. max (déclarée) [vert, pointillé] : max des probabilités verbalisées.
• Confiance déclarée brute [rose, pointillé] : le score de confiance brut \
que le modèle a directement exprimé.

Ce qu'on compare. Les traits pleins (logprobs) vs les traits pointillés \
(déclarée). On voit que les logprobs permettent un bien meilleur tri : \
en ne gardant que les 10-20% plus confiants, on atteint ~82-86% d'accuracy, \
alors que la confiance déclarée atteint ~50-72%. \
À couverture = 100% (toutes les questions), tout converge vers l'accuracy \
de base (~63% pour logprobs, ~45% pour déclarée — la différence vient du \
fait que les prédictions ne sont pas les mêmes).

Interprétation. Les logprobs sont nettement plus discriminants : le modèle \
"sait mieux qu'il dit" quand il a raison. La confiance verbalisée est mal \
calibrée et sépare mal les bonnes des mauvaises réponses."""
    ),
    (
        OUT_DIR / "fig2_conformal_prediction.png",
        "Figure 2 — Prédiction conforme : LAC vs APS",
        """\
Protocole. On utilise les mêmes 107 questions, séparées aléatoirement en \
54 de calibration et 53 de test. Les méthodes LAC et APS construisent des \
ensembles de prédiction {A}, {A,B}, {A,B,C}... au lieu d'une seule réponse. \
La garantie théorique est : P(vraie réponse ∈ ensemble) ≥ 1−α.

GRAPHE DE GAUCHE — Couverture vs α :
• Axe X — α (fixé par l'utilisateur) : le taux d'erreur nominal. Plus α \
est petit, plus on veut être sûr de couvrir la bonne réponse.
• Axe Y — Couverture empirique (calculée) : la fraction des questions test \
pour lesquelles la vraie réponse est bien dans l'ensemble prédit.
• Ligne grise pointillée : target = 1−α (la garantie théorique visée).
• Courbe LAC [bleu] : méthode Least Ambiguous set-valued Classifier. \
Seuil direct sur P(classe) ≥ 1−q̂.
• Courbe APS [orange] : Adaptive Prediction Sets. Ajoute les classes par \
probabilité décroissante jusqu'à atteindre le seuil q̂.

On observe que les deux méthodes respectent globalement la garantie (les \
courbes sont au-dessus de la ligne grise), sauf LAC qui sous-couvre \
légèrement pour α ≥ 0.15. APS est plus conservatrice (couverture plus \
élevée que le target).

GRAPHE DE DROITE — Taille d'ensemble vs α :
• Axe X — α (même paramètre fixé).
• Axe Y — Taille moyenne de l'ensemble prédit (calculée). Sur 4 classes \
possibles, la taille va de 1 (une seule réponse) à 4 (toutes les réponses).
• LAC produit des ensembles plus petits (1.5 à 3.2) qu'APS (2.3 à 4.0).

Interprétation. LAC est plus efficace (ensembles plus petits) mais \
peut sous-couvrir. APS est plus conservateur : il garantit mieux la \
couverture mais au prix d'ensembles plus grands. Sur seulement 107 \
questions (54 de calibration), les résultats sont bruités."""
    ),
    (
        OUT_DIR / "fig3_unified_comparison.png",
        "Figure 3 — Comparaison unifiée des 3 approches",
        """\
Ce graphe met sur le même repère les 3 familles de méthodes, même si \
elles ne mesurent pas exactement la même chose. L'objectif est de donner \
une vue d'ensemble.

Axe X — Couverture / Fraction retenue.
• Pour entropie et pmax : c'est la fraction des 107 questions retenues \
(triées par confiance décroissante).
• Pour conformal LAC : c'est la couverture empirique, i.e. P(y ∈ ensemble).

Axe Y — Accuracy / Target.
• Pour entropie et pmax : c'est l'accuracy des questions retenues.
• Pour conformal LAC : c'est le target 1−α correspondant.

Courbes :
• Entropie logprobs [bleu] et Prob. max logprobs [vert] : accuracy \
en fonction de la couverture (mêmes courbes que fig. 1, traits pleins).
• Confiance déclarée [rose] : accuracy en fonction de la couverture.
• Conformal LAC [orange, point-tiret] : pour chaque α de 0.01 à 0.50, \
on trace (couverture empirique, 1−α). Si la méthode est bien calibrée, \
les points devraient être au-dessus de la diagonale.

Interprétation. Les courbes logprobs (bleu/vert) dominent nettement la \
confiance déclarée (rose) à toutes les couvertures. La courbe conformal \
LAC monte en couverture vers la droite (les grands α donnent de petits \
ensembles, donc une couverture plus faible). On voit que la conformal \
prediction couvre bien quand on lui demande (couverture ≥ target pour \
la plupart des points).

Attention : les axes ne mesurent pas exactement la même chose pour les \
3 méthodes. Ce graphe est illustratif, pas une comparaison rigoureuse \
sur le même critère."""
    ),
    (
        OUT_DIR / "fig4_confidence_distributions.png",
        "Figure 4 — Distribution des scores : correctes vs incorrectes",
        """\
4 histogrammes montrant, pour chaque signal de confiance, comment les \
scores se distribuent selon que la prédiction est correcte (vert) ou \
incorrecte (rouge).

Axe X — Score de confiance (entre 0 et 1). C'est la valeur du signal \
pour chaque question.
Axe Y — Densité (normalisée). Les barres somment à 1 par couleur.

PANNEAU HAUT-GAUCHE — Entropie (logprobs).
Le score est 1 − entropie normalisée calculée sur la distribution softmax \
des logits. Les correctes (n=67) sont très concentrées vers 1.0 (le modèle \
est très sûr quand il a raison). Les incorrectes (n=40) sont plus étalées \
mais aussi assez concentrées vers les hauts scores → le modèle est souvent \
confiant même quand il se trompe (surconfiance), mais légèrement moins.

PANNEAU HAUT-DROIT — Prob. max (logprobs).
Le score est max(P(A), P(B), P(C), P(D)). Même constat, presque toutes \
les valeurs sont proches de 1.0, avec un pic encore plus marqué.

PANNEAU BAS-GAUCHE — Entropie (déclarée).
Même formule d'entropie mais sur les probabilités verbalisées. La \
distribution est beaucoup plus étalée. Les correctes et incorrectes se \
chevauchent fortement → ce signal discrimine très mal.

PANNEAU BAS-DROIT — Prob. max (déclarée).
La distribution est également étalée. On note que les correctes tendent \
vers les hauts scores (0.85-1.0) un peu plus que les incorrectes, mais \
le chevauchement reste important.

Interprétation. Les logprobs (panneaux du haut) concentrent les scores \
corrects vers 1.0, ce qui facilite le tri. La confiance déclarée \
(panneaux du bas) produit des distributions qui se chevauchent beaucoup : \
le modèle verbalise des confiances peu fiables. C'est la raison \
fondamentale de la supériorité des logprobs sur la figure 1."""
    ),
    (
        OUT_DIR / "fig5_logprob_vs_declared.png",
        "Figure 5 — Scatter : confiance logprobs vs confiance déclarée",
        """\
Chaque point représente une des 107 questions. On compare, question par \
question, le score de confiance issu des logprobs (axe Y) à celui issu \
de la confiance déclarée (axe X).

PANNEAU GAUCHE — Entropie.
• Axe X : confiance entropie déclarée = 1 − entropie normalisée des \
probabilités verbalisées.
• Axe Y : confiance entropie logprobs = 1 − entropie normalisée des \
probabilités softmax issues des logits.
• Points verts : prédiction correcte (selon les logprobs).
• Points rouges : prédiction incorrecte.
• Diagonale y=x : si les deux méthodes donnaient le même score.

On observe que la grande majorité des points sont au-dessus de la \
diagonale : les logprobs donnent des confiances systématiquement plus \
élevées que la confiance déclarée. Le modèle est internement plus \
"sûr de lui" (logprobs) qu'il ne le dit (déclarée). Les points verts \
sont concentrés en haut (haute confiance logprobs), tandis que les \
rouges sont plus éparpillés.

PANNEAU DROIT — Prob. max.
• Axe X : max des probabilités déclarées.
• Axe Y : max des probabilités logprobs.
Même constat amplifié : les logprobs sont quasi tous > 0.9 (écrasés \
en haut du graphe) car le softmax à température 0.5 est très piqué, \
alors que la confiance déclarée est plus variée (0.25 à 1.0).

Interprétation. Il y a un fort désaccord entre les deux sources de \
confiance. Les logprobs sont systématiquement plus extrêmes (proches \
de 0 ou 1) que les confiances verbalisées. Cela illustre le fait que \
l'auto-évaluation du modèle (déclarée) ne reflète pas bien ses \
probabilités internes (logprobs)."""
    ),
]


def wrap_text(text, width=95):
    """Wrap le texte en respectant les sauts de ligne explicites."""
    paragraphs = text.split("\n")
    wrapped = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(p, width=width))
    return "\n".join(wrapped)


def main():
    with PdfPages(str(PDF_PATH)) as pdf:
        # ── Page de titre ──
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.70, "Rapport explicatif", fontsize=28,
                 ha="center", va="center", fontweight="bold")
        fig.text(0.5, 0.60, "Comparaison des méthodes d'incertitude", fontsize=18,
                 ha="center", va="center")
        fig.text(0.5, 0.52, "MMLU — Gemini 2.0 Flash — 107 questions", fontsize=14,
                 ha="center", va="center", color="gray")
        fig.text(0.5, 0.38,
                 "Ce rapport présente chaque figure avec une explication détaillée :\n"
                 "• Ce que représentent les axes (fixé vs calculé)\n"
                 "• Ce que chaque courbe / symbole représente\n"
                 "• Comment interpréter les résultats\n\n"
                 "Méthodes comparées :\n"
                 "  1. Logprobs (scores de confiance issus des logits du modèle)\n"
                 "  2. Confiance déclarée (le modèle verbalise sa confiance)\n"
                 "  3. Prédiction conforme (LAC & APS — ensembles de prédiction)",
                 fontsize=10, ha="center", va="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#E3F2FD", alpha=0.8))
        pdf.savefig(fig)
        plt.close(fig)

        # ── Pages figure + explication ──
        for fig_path, title, explanation in FIGURES:
            if not fig_path.exists():
                print(f"  [skip] {fig_path}")
                continue

            # Page figure
            img = plt.imread(str(fig_path))
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0.5)
            pdf.savefig(fig)
            plt.close(fig)

            # Page explication
            wrapped = wrap_text(explanation, width=95)
            lines = wrapped.split("\n")

            # Découper en pages de ~45 lignes si l'explication est longue
            lines_per_page = 45
            for page_start in range(0, len(lines), lines_per_page):
                page_lines = lines[page_start:page_start + lines_per_page]
                text_block = "\n".join(page_lines)

                fig = plt.figure(figsize=(8.27, 11.69))
                if page_start == 0:
                    fig.text(0.5, 0.95, title, fontsize=13, ha="center",
                             va="top", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.4",
                                       facecolor="#1565C0", alpha=0.9),
                             color="white")
                    y_start = 0.90
                else:
                    y_start = 0.95

                fig.text(0.06, y_start, text_block,
                         fontsize=9.5, ha="left", va="top",
                         fontfamily="sans-serif",
                         linespacing=1.5,
                         transform=fig.transFigure)

                pdf.savefig(fig)
                plt.close(fig)

        # ── Page conclusion ──
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.85, "Conclusions", fontsize=20,
                 ha="center", va="center", fontweight="bold")

        conclusion = """\
1. Les logprobs (probabilités issues des logits) sont nettement supérieurs à la confiance
   déclarée (verbalisée) pour trier les réponses par fiabilité. En ne gardant que les 20%
   de questions les plus confiantes selon les logprobs, on atteint ~86% d'accuracy, contre
   ~50% avec la confiance déclarée.

2. L'entropie normalisée et la probabilité max donnent des résultats très similaires quand
   on utilise les logprobs (car le softmax à température 0.5 produit des distributions très
   piquées).

3. La confiance déclarée par le modèle est mal calibrée : elle ne reflète pas bien les
   probabilités internes. Il y a un fort désaccord entre confiance logprobs et déclarée
   (scatter plots).

4. La prédiction conforme (LAC, APS) offre des garanties théoriques de couverture qui sont
   globalement respectées. LAC produit des ensembles plus petits, APS est plus conservateur.
   Cependant, avec seulement 54 questions de calibration, les résultats sont bruités.

5. Limite : l'analyse porte sur 107 questions (celles avec logits A/B/C/D complets sur 500).
   Ce sous-ensemble n'est pas nécessairement représentatif de l'ensemble des questions MMLU."""

        fig.text(0.06, 0.75, conclusion,
                 fontsize=10.5, ha="left", va="top",
                 fontfamily="sans-serif", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF explicatif : {PDF_PATH}")


if __name__ == "__main__":
    main()
