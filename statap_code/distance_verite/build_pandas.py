#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construction d'une table d'interprétation des corrélations
entre scores de confiance et distance à la vérité.

Entrées :
    corr_all.csv
    corr_wrong_only.csv

Sortie :
    correlation_interpretation_table.csv

Objectif :
    Produire une table propre permettant d'interpréter
    les corrélations entre scores de confiance et mesures d'erreur,
    à la fois :
        - sur l'ensemble des réponses
        - sur les réponses fausses uniquement

Les scores sont considérés comme des signaux de confiance
et non comme des probabilités de correction.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PARAMÈTRES
# ============================================================

INPUT_CORR_ALL = "analysis_results/corr_all.csv"
INPUT_CORR_WRONG = "analysis_results/corr_wrong_only.csv"

OUTPUT_FILE = "correlation_interpretation_table.csv"


# ============================================================
# CLASSIFICATION DES SCORES
# ============================================================

SCORE_TYPE_MAP = {

    # scores explicites
    "confidence_verbalized": "model_confidence",
    "judge_confidence": "judge_confidence",

    # tokens de la réponse
    "prob_tokens_only.prob_joint": "token_probability",
    "prob_tokens_only.prob_geo_mean": "token_probability",
    "prob_tokens_only.logprob_sum": "token_logscore",
    "prob_tokens_only.logprob_avg": "token_logscore",
    "prob_tokens_only.perplexity": "token_uncertainty",
    "prob_tokens_only.p_min": "token_probability",

    # tous les tokens
    "prob_all_tokens.prob_joint": "sequence_probability",
    "prob_all_tokens.prob_geo_mean": "sequence_probability",
    "prob_all_tokens.logprob_sum": "sequence_logscore",
    "prob_all_tokens.logprob_avg": "sequence_logscore",
    "prob_all_tokens.perplexity": "sequence_uncertainty",
    "prob_all_tokens.p_min": "sequence_probability",
}


# signe attendu de la corrélation avec l'erreur
# True = corrélation négative attendue
# False = corrélation positive attendue

EXPECTED_NEGATIVE = {

    "confidence_verbalized": True,
    "judge_confidence": True,

    "prob_tokens_only.prob_joint": True,
    "prob_tokens_only.prob_geo_mean": True,
    "prob_tokens_only.logprob_sum": True,
    "prob_tokens_only.logprob_avg": True,
    "prob_tokens_only.perplexity": False,
    "prob_tokens_only.p_min": True,

    "prob_all_tokens.prob_joint": True,
    "prob_all_tokens.prob_geo_mean": True,
    "prob_all_tokens.logprob_sum": True,
    "prob_all_tokens.logprob_avg": True,
    "prob_all_tokens.perplexity": False,
    "prob_all_tokens.p_min": True,
}


# ============================================================
# FONCTIONS D'INTERPRÉTATION
# ============================================================

def effect_size_label(rho):
    """Interprétation de la magnitude de Spearman"""
    if pd.isna(rho):
        return "NA"

    a = abs(rho)

    if a < 0.05:
        return "quasi nul"
    elif a < 0.10:
        return "très faible"
    elif a < 0.20:
        return "faible"
    elif a < 0.30:
        return "faible à modéré"
    elif a < 0.50:
        return "modéré"
    else:
        return "fort"


def sign_direction_label(score_col, rho):
    """Vérifie si le signe est celui attendu"""

    if pd.isna(rho):
        return "NA"

    expected_negative = EXPECTED_NEGATIVE.get(score_col, True)

    if expected_negative:

        if rho < 0:
            return "sens attendu"

        elif rho > 0:
            return "sens opposé"

        else:
            return "nul"

    else:

        if rho > 0:
            return "sens attendu"

        elif rho < 0:
            return "sens opposé"

        else:
            return "nul"


def interpretation_text(score_col, rho, scope):

    if pd.isna(rho):
        return "corrélation non disponible"

    effect = effect_size_label(rho)
    direction = sign_direction_label(score_col, rho)

    if scope == "all":
        base = "Sur l’ensemble des réponses"
    else:
        base = "Parmi les réponses fausses uniquement"

    if direction == "sens attendu":

        if EXPECTED_NEGATIVE.get(score_col, True):

            return (
                f"{base}, le score présente un lien {effect} avec la distance à la vérité : "
                f"les réponses associées à une confiance plus élevée tendent à être plus proches de la valeur correcte."
            )

        else:

            return (
                f"{base}, le score présente un lien {effect} avec la distance à la vérité : "
                f"une perplexité plus élevée est associée à des erreurs plus importantes."
            )

    elif direction == "sens opposé":

        return (
            f"{base}, le score présente un lien {effect} mais dans le sens opposé à celui attendu."
        )

    else:

        return (
            f"{base}, aucun lien clair n'apparaît entre ce score et la distance à la vérité."
        )


# ============================================================
# CONSTRUCTION TABLE FINALE
# ============================================================

def build_interpretation_table(corr_all_df, corr_wrong_df):

    all_df = corr_all_df.rename(columns={
        "pearson": "pearson_all",
        "spearman": "spearman_all",
        "n": "n_all",
    })

    wrong_df = corr_wrong_df.rename(columns={
        "pearson_wrong_only": "pearson_wrong_only",
        "spearman_wrong_only": "spearman_wrong_only",
        "n_wrong": "n_wrong_only",
    })

    merged = all_df.merge(
        wrong_df,
        on=["confidence_col", "error_col"],
        how="outer"
    )

    merged["score_type"] = merged["confidence_col"].map(SCORE_TYPE_MAP).fillna("other")

    merged["expected_direction"] = merged["confidence_col"].map(
        lambda c: "negative" if EXPECTED_NEGATIVE.get(c, True) else "positive"
    )

    merged["effect_all"] = merged["spearman_all"].apply(effect_size_label)
    merged["effect_wrong_only"] = merged["spearman_wrong_only"].apply(effect_size_label)

    merged["direction_all"] = merged.apply(
        lambda row: sign_direction_label(row["confidence_col"], row["spearman_all"]),
        axis=1
    )

    merged["direction_wrong_only"] = merged.apply(
        lambda row: sign_direction_label(row["confidence_col"], row["spearman_wrong_only"]),
        axis=1
    )

    merged["interpretation_all"] = merged.apply(
        lambda row: interpretation_text(row["confidence_col"], row["spearman_all"], "all"),
        axis=1
    )

    merged["interpretation_wrong_only"] = merged.apply(
        lambda row: interpretation_text(row["confidence_col"], row["spearman_wrong_only"], "wrong_only"),
        axis=1
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print("Lecture des fichiers de corrélation...")

    corr_all_df = pd.read_csv(INPUT_CORR_ALL)
    corr_wrong_df = pd.read_csv(INPUT_CORR_WRONG)

    print("Construction de la table d'interprétation...")

    table = build_interpretation_table(corr_all_df, corr_wrong_df)

    table.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Table créée :", OUTPUT_FILE)
    print()
    print("Aperçu :")
    print(table.head(20))


if __name__ == "__main__":
    main()