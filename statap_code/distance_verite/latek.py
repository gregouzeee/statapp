import pandas as pd

# ============================================================
# Chargement
# ============================================================

df = pd.read_csv("analysis_results/calibration_table.csv")

score_names = {
    "confidence_verbalized": "Conf verbalisée",
    "judge_confidence": "Conf judge",

    "prob_tokens_only.prob_joint": "Prob joint seq",
    "prob_tokens_only.prob_geo_mean": "Prob mean seq",

    "prob_all_tokens.prob_joint": "Prob joint all",
    "prob_all_tokens.prob_geo_mean": "Prob mean all",
    "prob_all_tokens.p_min": "Prob min all",
}

df["confidence_col"] = df["confidence_col"].replace(score_names)

# ============================================================
# Garder uniquement les scores avec calibration
# ============================================================

df = df[df["brier"].notna()].copy()

df = df[[
    "confidence_col",
    "brier",
    "ece_10bins"
]]

df = df.rename(columns={
    "confidence_col": "Confidence score",
    "brier": "Brier",
    "ece_10bins": "ECE"
})

# ============================================================
# Export LaTeX
# ============================================================

latex = df.to_latex(
    index=False,
    escape=True,
    longtable=True,
    caption="Indicateurs de calibration des scores probabilistes",
    label="tab:confidence_calibration",
    column_format="lrr",
    float_format="%.6f"
)

with open("confidence_calibration_table.tex", "w", encoding="utf-8") as f:
    f.write(latex)

print(latex)