from pathlib import Path
import pandas as pd

INPUT_DIR = Path("statap_code/gsm8k/analysis_outputs/")
OUTPUT_DIR = Path("./latex_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# Helpers
# =========================================================

def prettify_feature_name(name: str) -> str:
    """
    Rend les noms de variables plus lisibles dans les tableaux.
    """
    mapping = {
        "all_": "Full generation: ",
        "reasoning_": "Reasoning: ",
        "final_": "Final answer: ",
        "_last_third_mean": "mean on last third",
        "_first_third_mean": "mean on first third",
        "_last_minus_first_third_mean": "last third - first third",
        "_entropy_mean": "entropy mean",
        "_entropy_std": "entropy std",
        "_entropy_max": "entropy max",
        "_top1_last_third_mean": "top1 mean on last third",
        "_top1_min": "top1 min",
        "_top1_slope": "top1 slope",
        "_margin_mean": "margin mean",
        "_margin_min": "margin min",
        "_length": "length",
        "_longest_run_lp_below_-2": "longest run with logprob < -2",
    }

    out = name
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str,
                      column_format: str = None, float_format="%.3f"):
    latex = df.to_latex(
        index=False,
        escape=False,
        float_format=float_format,
        column_format=column_format
    )

    wrapped = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )

    path.write_text(wrapped, encoding="utf-8")


# =========================================================
# 1. Model summary table
# =========================================================

models = pd.read_csv(INPUT_DIR / "logit_model_summary.csv")

models = models[[
    "model_name",
    "n_features",
    "cv_roc_auc",
    "cv_average_precision"
]].copy()

models = models.sort_values("cv_roc_auc", ascending=False)

model_name_map = {
    "all_numeric": "All variables",
    "all_numeric_no_length": "All variables (without length)",
    "top_manual": "Interpretable subset",
    "reasoning_only": "Reasoning only",
    "reasoning_no_length": "Reasoning only (without length)",
    "all_only": "Full-generation only",
    "final_only": "Final-answer only",
    "all_no_length": "Full-generation only (without length)",
}

models["model_name"] = models["model_name"].map(model_name_map).fillna(models["model_name"])

models = models.rename(columns={
    "model_name": "Feature set",
    "n_features": "Nb. features",
    "cv_roc_auc": "ROC-AUC",
    "cv_average_precision": "Average Precision",
})

write_latex_table(
    models,
    OUTPUT_DIR / "logprob_model_summary_table.tex",
    caption="Performance des différents ensembles de variables pour la détection des erreurs.",
    label="tab:logprob_model_summary",
    column_format="lccc"
)


# =========================================================
# 2. Top univariate features table
# =========================================================

uni = pd.read_csv(INPUT_DIR / "top15_univariate_features.csv")

uni_small = uni[[
    "feature",
    "mean_correct",
    "mean_wrong",
    "univariate_auc",
    "mannwhitney_fdr_bh"
]].copy()

uni_small = uni_small.head(10)

uni_small["feature"] = uni_small["feature"].apply(prettify_feature_name)

uni_small = uni_small.rename(columns={
    "feature": "Variable",
    "mean_correct": "Mean correct",
    "mean_wrong": "Mean wrong",
    "univariate_auc": "AUC",
    "mannwhitney_fdr_bh": "FDR-BH",
})

write_latex_table(
    uni_small,
    OUTPUT_DIR / "logprob_top_univariate_table.tex",
    caption="Variables univariées les plus discriminantes entre réponses correctes et incorrectes.",
    label="tab:logprob_top_univariate",
    column_format="lcccc"
)


# =========================================================
# 3. Optional: coefficients of interpretable logistic model
# =========================================================

coef = pd.read_csv(INPUT_DIR / "logit_coefficients_top_manual.csv").copy()
coef["feature"] = coef["feature"].apply(prettify_feature_name)

coef = coef.rename(columns={
    "feature": "Variable",
    "coef": "Coefficient",
})

write_latex_table(
    coef,
    OUTPUT_DIR / "logprob_top_manual_coefficients_table.tex",
    caption="Coefficients du modèle logistique construit sur le sous-ensemble interprétable de variables.",
    label="tab:logprob_top_manual_coefficients",
    column_format="lc"
)

print("Tables écrites dans :", OUTPUT_DIR.resolve())