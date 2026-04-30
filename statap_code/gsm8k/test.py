import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_CSV = "statap_code/gsm8k/logprob_features_interpretable.csv"
OUTPUT_DIR = Path("statap_code/gsm8k/analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Load data
# =========================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "correct" not in df.columns:
        raise ValueError("La colonne 'correct' est absente du CSV.")

    df["correct"] = df["correct"].astype(int)
    return df


# =========================================================
# 2. Utilities
# =========================================================

def cliffs_delta(x, y):
    """
    Effet de taille non paramétrique.
    Retourne une valeur dans [-1, 1].
    Positif si x tend à être plus grand que y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return (gt - lt) / (len(x) * len(y))


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    pooled = np.sqrt(((len(x) - 1) * sx**2 + (len(y) - 1) * sy**2) / (len(x) + len(y) - 2))
    if pooled == 0:
        return np.nan

    return (x.mean() - y.mean()) / pooled


def benjamini_hochberg(pvals):
    """
    Correction BH-FDR.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def infer_auc_direction(y, scores):
    """
    Pour chaque feature, on choisit le sens qui donne la meilleure AUC :
    - score direct
    - score négatif
    Retourne auc et direction retenue.
    """
    auc_direct = roc_auc_score(y, scores)
    auc_inverse = roc_auc_score(y, -scores)

    if auc_direct >= auc_inverse:
        return auc_direct, "higher_is_more_correct"
    return auc_inverse, "lower_is_more_correct"


# =========================================================
# 3. Univariate analysis
# =========================================================

def get_numeric_features(df: pd.DataFrame):
    excluded = {"row_index", "id", "correct", "group"}
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    y = df["correct"].values
    features = get_numeric_features(df)

    rows = []

    correct_df = df[df["correct"] == 1]
    wrong_df = df[df["correct"] == 0]

    for col in features:
        x1 = correct_df[col].dropna().values
        x0 = wrong_df[col].dropna().values

        if len(x1) < 3 or len(x0) < 3:
            continue

        mean_correct = float(np.mean(x1))
        mean_wrong = float(np.mean(x0))
        median_correct = float(np.median(x1))
        median_wrong = float(np.median(x0))
        std_correct = float(np.std(x1, ddof=1)) if len(x1) > 1 else np.nan
        std_wrong = float(np.std(x0, ddof=1)) if len(x0) > 1 else np.nan

        # Welch t-test
        try:
            t_stat, t_p = ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
        except Exception:
            t_stat, t_p = np.nan, np.nan

        # Mann-Whitney
        try:
            mw_stat, mw_p = mannwhitneyu(x1, x0, alternative="two-sided")
        except Exception:
            mw_stat, mw_p = np.nan, np.nan

        # AUC univariée
        valid = df[["correct", col]].dropna()
        auc, direction = infer_auc_direction(valid["correct"].values, valid[col].values)

        rows.append({
            "feature": col,
            "n_correct": len(x1),
            "n_wrong": len(x0),
            "mean_correct": mean_correct,
            "mean_wrong": mean_wrong,
            "diff_wrong_minus_correct": mean_wrong - mean_correct,
            "median_correct": median_correct,
            "median_wrong": median_wrong,
            "std_correct": std_correct,
            "std_wrong": std_wrong,
            "welch_t_pvalue": t_p,
            "mannwhitney_pvalue": mw_p,
            "cohens_d_correct_minus_wrong": cohens_d(x1, x0),
            "cliffs_delta_correct_vs_wrong": cliffs_delta(x1, x0),
            "univariate_auc": auc,
            "auc_direction": direction,
        })

    res = pd.DataFrame(rows)

    if len(res) > 0:
        res["welch_t_fdr_bh"] = benjamini_hochberg(res["welch_t_pvalue"].fillna(1.0).values)
        res["mannwhitney_fdr_bh"] = benjamini_hochberg(res["mannwhitney_pvalue"].fillna(1.0).values)
        res = res.sort_values(
            ["univariate_auc", "mannwhitney_fdr_bh"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return res


# =========================================================
# 4. Logistic regression with CV
# =========================================================

def evaluate_logistic_regression(df: pd.DataFrame, features, model_name: str):
    X = df[features].copy()
    y = df["correct"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            penalty="l2"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Probabilités hors-échantillon
    y_proba = cross_val_predict(
        pipe, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    auc = roc_auc_score(y, y_proba)
    ap = average_precision_score(y, y_proba)

    # Fit final sur tout l'échantillon pour inspecter les coefficients
    pipe.fit(X, y)
    clf = pipe.named_steps["clf"]

    coef_df = pd.DataFrame({
        "feature": features,
        "coef": clf.coef_[0]
    }).sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True)

    metrics = {
        "model_name": model_name,
        "n_features": len(features),
        "cv_roc_auc": auc,
        "cv_average_precision": ap,
        "class_balance_correct_rate": float(np.mean(y)),
    }

    return metrics, coef_df, y_proba


# =========================================================
# 5. Feature sets
# =========================================================

def build_feature_sets(df: pd.DataFrame):
    all_numeric = get_numeric_features(df)

    reasoning_feats = [c for c in all_numeric if c.startswith("reasoning_")]
    final_feats = [c for c in all_numeric if c.startswith("final_")]
    all_feats = [c for c in all_numeric if c.startswith("all_")]

    length_feats = [c for c in all_numeric if c.endswith("_length")]
    no_length = [c for c in all_numeric if not c.endswith("_length")]

    top_manual = [
        "reasoning_length",
        "all_length",
        "reasoning_margin_mean",
        "all_margin_mean",
        "final_margin_mean",
        "reasoning_top1_min",
        "reasoning_margin_min",
        "reasoning_entropy_max",
        "reasoning_longest_run_lp_below_-2",
        "reasoning_entropy_last_minus_first_third_mean",
        "reasoning_entropy_mean",
    ]
    top_manual = [c for c in top_manual if c in df.columns]

    feature_sets = {
        "reasoning_only": reasoning_feats,
        "final_only": final_feats,
        "all_only": all_feats,
        "all_numeric": all_numeric,
        "all_numeric_no_length": no_length,
        "top_manual": top_manual,
    }

    # un set sans les variables de longueur
    feature_sets["reasoning_no_length"] = [c for c in reasoning_feats if c not in length_feats]
    feature_sets["all_no_length"] = [c for c in all_feats if c not in length_feats]

    return feature_sets


# =========================================================
# 6. Save helpers
# =========================================================

def save_dataframe(df: pd.DataFrame, name: str):
    csv_path = OUTPUT_DIR / f"{name}.csv"
    html_path = OUTPUT_DIR / f"{name}.html"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>{name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            .table-container {{
                width: 100%;
                overflow-x: auto;
                max-height: 85vh;
                overflow-y: auto;
                border: 1px solid #ccc;
            }}
            table {{
                border-collapse: collapse;
                min-width: 1200px;
                width: max-content;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 6px 10px;
                white-space: nowrap;
                text-align: right;
            }}
            th {{
                background: #f5f5f5;
                position: sticky;
                top: 0;
            }}
            td:first-child, th:first-child {{
                text-align: left;
                position: sticky;
                left: 0;
                background: #fafafa;
            }}
        </style>
    </head>
    <body>
        <h1>{name}</h1>
        <div class="table-container">
            {df.to_html(index=False, float_format=lambda x: f"{x:.6f}")}
        </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")


# =========================================================
# 7. Main
# =========================================================

def main():
    df = load_data(INPUT_CSV)

    print("\n==============================")
    print("DATASET")
    print("==============================")
    print(df["correct"].value_counts(dropna=False))
    print(f"Accuracy = {df['correct'].mean():.4f}")

    # ---------------------------------
    # A. Univariate tests
    # ---------------------------------
    uni = univariate_analysis(df)
    save_dataframe(uni, "univariate_tests")

    print("\n==============================")
    print("TOP UNIVARIATE FEATURES")
    print("==============================")
    cols_to_show = [
        "feature",
        "mean_correct",
        "mean_wrong",
        "diff_wrong_minus_correct",
        "mannwhitney_pvalue",
        "mannwhitney_fdr_bh",
        "univariate_auc",
        "auc_direction",
        "cohens_d_correct_minus_wrong",
        "cliffs_delta_correct_vs_wrong",
    ]
    print(uni[cols_to_show].head(20).round(4))

    # ---------------------------------
    # B. Logistic models
    # ---------------------------------
    feature_sets = build_feature_sets(df)

    model_rows = []
    all_coef_tables = {}

    for name, feats in feature_sets.items():
        feats = [c for c in feats if c in df.columns]
        if len(feats) == 0:
            continue

        metrics, coef_df, y_proba = evaluate_logistic_regression(df, feats, name)
        model_rows.append(metrics)
        all_coef_tables[name] = coef_df

        save_dataframe(coef_df, f"logit_coefficients_{name}")

    model_summary = pd.DataFrame(model_rows).sort_values("cv_roc_auc", ascending=False)
    save_dataframe(model_summary, "logit_model_summary")

    print("\n==============================")
    print("LOGISTIC REGRESSION SUMMARY")
    print("==============================")
    print(model_summary.round(4))

    # ---------------------------------
    # C. Compare top univariate vs logit
    # ---------------------------------
    best_uni = uni.head(15).copy()
    save_dataframe(best_uni, "top15_univariate_features")

    # Save a compact JSON summary
    summary = {
        "n_rows": int(len(df)),
        "n_correct": int((df["correct"] == 1).sum()),
        "n_wrong": int((df["correct"] == 0).sum()),
        "accuracy": float(df["correct"].mean()),
        "best_univariate_feature": None if len(uni) == 0 else uni.iloc[0]["feature"],
        "best_univariate_auc": None if len(uni) == 0 else float(uni.iloc[0]["univariate_auc"]),
        "best_logit_model": None if len(model_summary) == 0 else model_summary.iloc[0]["model_name"],
        "best_logit_auc": None if len(model_summary) == 0 else float(model_summary.iloc[0]["cv_roc_auc"]),
    }

    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==============================")
    print("FILES WRITTEN")
    print("==============================")
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(p)


if __name__ == "__main__":
    main()