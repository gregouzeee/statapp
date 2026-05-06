#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse complète d'un fichier JSONL de réponses numériques avec scores de confiance.

Entrée attendue : un fichier .jsonl dont chaque ligne est un dict du type :
{
  "question": "...",
  "answer_true": 1914.0,
  "answer_model": 1914.0,
  "confidence_verbalized": 0.95,
  "prob_all_tokens": {...},
  "prob_tokens_only": {...},
  "judge_confidence": 0.9,
  "abs_error": 0.0,
  "rel_error": 0.0,
  "log_error": 0.0
}

Sorties :
- data_flattened.csv
- descriptive_stats.csv
- score_summary.csv
- corr_all.csv
- corr_wrong_only.csv
- auc_table.csv
- calibration_table.csv
- logistic_univariate.csv
- logistic_multivariate.csv
- linear_wrong_only.csv
- summary.json
- figures PNG (histogrammes, reliability plots, scatterplots)

Usage :
python analyze_confidence.py --jsonl data.jsonl --outdir results
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# IO / flatten
# ============================================================

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON invalide à la ligne {i}: {e}") from e
    return rows


def flatten_record(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, dict) and k in ("prob_all_tokens", "prob_tokens_only"):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def to_numeric_safe(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df


# ============================================================
# Utils stats
# ============================================================

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def is_probability_like_series(s: pd.Series, tol: float = 1e-9) -> bool:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return False
    return bool(((x >= -tol) & (x <= 1 + tol)).all())


def safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def quantile_bins(probs: np.ndarray, n_bins: int = 10) -> np.ndarray:
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(probs, q)
    edges[0] = min(edges[0], probs.min())
    edges[-1] = max(edges[-1], probs.max())
    # éviter les bords identiques
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    idx = np.digitize(probs, edges[1:-1], right=True)
    return idx


def bootstrap_stat_xy(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    min_valid: int = 200,
) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            val = stat_fn(x[idx], y[idx])
            if np.isfinite(val):
                vals.append(val)
        except Exception:
            continue

    if len(vals) < min_valid:
        return np.nan, np.nan, len(vals)

    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi, len(vals)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        raise ValueError("AUC impossible: une seule classe.")
    return float(roc_auc_score(y, score))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# Préparation des données
# ============================================================

def build_derived_columns(df: pd.DataFrame, eps_rel: float = 1e-8) -> pd.DataFrame:
    if "abs_error" not in df.columns and {"answer_true", "answer_model"}.issubset(df.columns):
        df["abs_error"] = (pd.to_numeric(df["answer_true"], errors="coerce") -
                           pd.to_numeric(df["answer_model"], errors="coerce")).abs()

    if "rel_error" not in df.columns and {"answer_true", "answer_model"}.issubset(df.columns):
        yt = pd.to_numeric(df["answer_true"], errors="coerce").abs()
        yp = pd.to_numeric(df["answer_model"], errors="coerce")
        df["rel_error"] = (yp - pd.to_numeric(df["answer_true"], errors="coerce")).abs() / (yt + eps_rel)

    if "log_error" not in df.columns and {"answer_true", "answer_model"}.issubset(df.columns):
        yt = pd.to_numeric(df["answer_true"], errors="coerce")
        yp = pd.to_numeric(df["answer_model"], errors="coerce")
        mask = (yt > 0) & (yp > 0)
        df["log_error"] = np.nan
        df.loc[mask, "log_error"] = np.abs(np.log(yp[mask]) - np.log(yt[mask]))

    if "abs_error" in df.columns:
        df["is_correct"] = (pd.to_numeric(df["abs_error"], errors="coerce") == 0).astype(int)
        df["log1p_abs_error"] = np.log1p(pd.to_numeric(df["abs_error"], errors="coerce"))
    else:
        df["is_correct"] = np.nan
        df["log1p_abs_error"] = np.nan

    return df


def infer_confidence_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "confidence_verbalized",
        "judge_confidence",
        "prob_tokens_only.prob_joint",
        "prob_tokens_only.prob_geo_mean",
        "prob_tokens_only.logprob_sum",
        "prob_tokens_only.logprob_avg",
        "prob_tokens_only.perplexity",
        "prob_tokens_only.p_min",
        "prob_all_tokens.prob_joint",
        "prob_all_tokens.prob_geo_mean",
        "prob_all_tokens.logprob_sum",
        "prob_all_tokens.logprob_avg",
        "prob_all_tokens.perplexity",
        "prob_all_tokens.p_min",
    ]
    existing = [c for c in preferred if c in df.columns and is_numeric_series(df[c])]
    return existing


# ============================================================
# Descriptif
# ============================================================

def descriptive_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "column": c,
            "n_non_null": int(s.notna().sum()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "probability_like_0_1": is_probability_like_series(s),
        })
    return pd.DataFrame(rows)


# ============================================================
# Corrélations
# ============================================================

def correlation_tables(
    df: pd.DataFrame,
    conf_cols: List[str],
    error_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_all = []
    rows_wrong = []

    for c in conf_cols:
        for e in error_cols:
            if c not in df.columns or e not in df.columns:
                continue

            tmp = df[[c, e]].dropna()
            if len(tmp) >= 5:
                x = pd.to_numeric(tmp[c], errors="coerce").to_numpy()
                y = pd.to_numeric(tmp[e], errors="coerce").to_numpy()
                rows_all.append({
                    "confidence_col": c,
                    "error_col": e,
                    "n": len(tmp),
                    "pearson": safe_pearson(x, y),
                    "spearman": safe_spearman(x, y),
                })

            tmpw = df.loc[df["is_correct"] == 0, [c, e]].dropna()
            if len(tmpw) >= 5:
                xw = pd.to_numeric(tmpw[c], errors="coerce").to_numpy()
                yw = pd.to_numeric(tmpw[e], errors="coerce").to_numpy()
                rows_wrong.append({
                    "confidence_col": c,
                    "error_col": e,
                    "n_wrong": len(tmpw),
                    "pearson_wrong_only": safe_pearson(xw, yw),
                    "spearman_wrong_only": safe_spearman(xw, yw),
                })

    return pd.DataFrame(rows_all), pd.DataFrame(rows_wrong)


# ============================================================
# AUC / calibration univariée
# ============================================================

def auc_analysis(df: pd.DataFrame, conf_cols: List[str], seed: int = 0) -> pd.DataFrame:
    rows = []

    for c in conf_cols:
        tmp = df[[c, "is_correct"]].dropna()
        if len(tmp) < 10:
            continue

        y = tmp["is_correct"].astype(int).to_numpy()
        score = pd.to_numeric(tmp[c], errors="coerce").to_numpy()

        if len(np.unique(y)) < 2:
            continue

        try:
            auc = safe_auc(y, score)
            lo, hi, n_valid = bootstrap_stat_xy(
                score,
                y,
                stat_fn=lambda s, yy: safe_auc(yy, s),
                n_boot=2000,
                alpha=0.05,
                seed=seed,
            )
        except Exception:
            auc, lo, hi, n_valid = np.nan, np.nan, np.nan, 0

        rows.append({
            "confidence_col": c,
            "n": len(tmp),
            "accuracy": float(y.mean()),
            "score_mean": float(np.mean(score)),
            "AUC": auc,
            "AUC_CI_lo": lo,
            "AUC_CI_hi": hi,
            "bootstrap_valid_samples": n_valid,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values("AUC", ascending=False)
    return out


def calibration_analysis(df: pd.DataFrame, conf_cols: List[str]) -> pd.DataFrame:
    rows = []

    for c in conf_cols:
        tmp = df[[c, "is_correct"]].dropna()
        if len(tmp) < 20:
            continue

        p = pd.to_numeric(tmp[c], errors="coerce").to_numpy()
        y = tmp["is_correct"].astype(int).to_numpy()

        if not is_probability_like_series(pd.Series(p)):
            rows.append({
                "confidence_col": c,
                "n": len(tmp),
                "is_probability_like": False,
                "brier": np.nan,
                "ece_10bins": np.nan,
                "mean_conf": float(np.mean(p)),
                "accuracy": float(np.mean(y)),
                "calib_intercept": np.nan,
                "calib_slope": np.nan,
            })
            continue

        p = np.clip(p, 1e-12, 1 - 1e-12)

        brier = brier_score_loss(y, p)
        ece = ece_score(y, p, n_bins=10)

        try:
            z = safe_logit(p).reshape(-1, 1)
            clf = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
            clf.fit(z, y)
            intercept = float(clf.intercept_[0])
            slope = float(clf.coef_.ravel()[0])
        except Exception:
            intercept = np.nan
            slope = np.nan

        rows.append({
            "confidence_col": c,
            "n": len(tmp),
            "is_probability_like": True,
            "brier": float(brier),
            "ece_10bins": float(ece),
            "mean_conf": float(np.mean(p)),
            "accuracy": float(np.mean(y)),
            "calib_intercept": intercept,
            "calib_slope": slope,
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values(["is_probability_like", "brier"], ascending=[False, True])
    return out


# ============================================================
# Régressions
# ============================================================

def logistic_univariate(df: pd.DataFrame, conf_cols: List[str]) -> pd.DataFrame:
    rows = []

    for c in conf_cols:
        tmp = df[[c, "is_correct"]].dropna()
        if len(tmp) < 30 or len(tmp["is_correct"].unique()) < 2:
            continue

        X = tmp[[c]].astype(float).to_numpy()
        y = tmp["is_correct"].astype(int).to_numpy()

        try:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ])
            model.fit(X, y)
            proba = model.predict_proba(X)[:, 1]
            auc = safe_auc(y, proba)
            brier = brier_score_loss(y, np.clip(proba, 1e-12, 1 - 1e-12))
            coef = float(model.named_steps["clf"].coef_.ravel()[0])
            intercept = float(model.named_steps["clf"].intercept_[0])

            rows.append({
                "confidence_col": c,
                "n": len(tmp),
                "coef_std": coef,
                "intercept": intercept,
                "train_auc": auc,
                "train_brier": brier,
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values("train_auc", ascending=False)
    return out


def logistic_multivariate(df: pd.DataFrame, conf_cols: List[str], seed: int = 0) -> pd.DataFrame:
    usable = [c for c in conf_cols if c in df.columns and is_numeric_series(df[c])]
    if len(usable) == 0:
        return pd.DataFrame()

    tmp = df[usable + ["is_correct"]].dropna()
    if len(tmp) < 50 or len(tmp["is_correct"].unique()) < 2:
        return pd.DataFrame()

    X = tmp[usable].astype(float).to_numpy()
    y = tmp["is_correct"].astype(int).to_numpy()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ])
        model.fit(X_train, y_train)

        p_train = model.predict_proba(X_train)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        coefs = model.named_steps["clf"].coef_.ravel()

        rows = []
        for feat, coef in zip(usable, coefs):
            rows.append({
                "feature": feat,
                "coef_std": float(coef),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_auc": safe_auc(y_train, p_train),
                "test_auc": safe_auc(y_test, p_test),
                "train_brier": brier_score_loss(y_train, np.clip(p_train, 1e-12, 1 - 1e-12)),
                "test_brier": brier_score_loss(y_test, np.clip(p_test, 1e-12, 1 - 1e-12)),
            })

        out = pd.DataFrame(rows).sort_values("coef_std", ascending=False)
        return out
    except Exception:
        return pd.DataFrame()


def linear_wrong_only(df: pd.DataFrame, conf_cols: List[str], seed: int = 0) -> pd.DataFrame:
    usable = [c for c in conf_cols if c in df.columns and is_numeric_series(df[c])]
    if len(usable) == 0:
        return pd.DataFrame()

    tmp = df.loc[df["is_correct"] == 0, usable + ["log1p_abs_error"]].dropna()
    if len(tmp) < 30:
        return pd.DataFrame()

    X = tmp[usable].astype(float).to_numpy()
    y = tmp["log1p_abs_error"].astype(float).to_numpy()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ])
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        coefs = model.named_steps["reg"].coef_

        rows = []
        for feat, coef in zip(usable, coefs):
            rows.append({
                "feature": feat,
                "coef_std": float(coef),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "train_rmse": mean_squared_error(y_train, y_train_pred) ** 0.5,
                "test_rmse": mean_squared_error(y_test, y_test_pred) ** 0.5,
                "train_r2": r2_score(y_train, y_train_pred),
                "test_r2": r2_score(y_test, y_test_pred),
            })

        out = pd.DataFrame(rows).sort_values("coef_std", ascending=False)
        return out
    except Exception:
        return pd.DataFrame()


# ============================================================
# Figures
# ============================================================

def save_histogram(series: pd.Series, title: str, outpath: Path) -> None:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(x) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=30)
    plt.title(title)
    plt.xlabel("Valeur")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_boxplot_by_correct(df: pd.DataFrame, col: str, outpath: Path) -> None:
    tmp = df[[col, "is_correct"]].dropna()
    if len(tmp) == 0:
        return

    vals0 = tmp.loc[tmp["is_correct"] == 0, col].astype(float).to_numpy()
    vals1 = tmp.loc[tmp["is_correct"] == 1, col].astype(float).to_numpy()

    if len(vals0) == 0 and len(vals1) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.boxplot([vals0, vals1], tick_labels=["Incorrect", "Correct"])
    plt.title(f"{col} selon exactitude")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def reliability_table(df: pd.DataFrame, conf_col: str, n_bins: int = 10) -> pd.DataFrame:
    tmp = df[[conf_col, "is_correct"]].dropna()
    if len(tmp) < 20:
        return pd.DataFrame()

    p = pd.to_numeric(tmp[conf_col], errors="coerce").to_numpy()
    y = tmp["is_correct"].astype(int).to_numpy()

    if not is_probability_like_series(pd.Series(p)):
        return pd.DataFrame()

    p = np.clip(p, 0, 1)
    bin_id = quantile_bins(p, n_bins=n_bins)

    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_pred": float(p[mask].mean()),
            "empirical_acc": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def save_reliability_plot(tab: pd.DataFrame, title: str, outpath: Path) -> None:
    if len(tab) == 0:
        return
    plt.figure(figsize=(5, 5))
    plt.plot(tab["mean_pred"], tab["empirical_acc"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confiance moyenne prédite")
    plt.ylabel("Précision empirique")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_scatter_wrong(df: pd.DataFrame, conf_col: str, err_col: str, outpath: Path) -> None:
    tmp = df.loc[df["is_correct"] == 0, [conf_col, err_col]].dropna()
    if len(tmp) < 5:
        return

    x = pd.to_numeric(tmp[conf_col], errors="coerce").to_numpy()
    y = pd.to_numeric(tmp[err_col], errors="coerce").to_numpy()

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=10, alpha=0.3)
    plt.xlabel(conf_col)
    plt.ylabel(err_col)
    plt.title(f"{err_col} vs {conf_col} (réponses fausses)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================================================
# Résumé global
# ============================================================

def build_summary_json(
    df: pd.DataFrame,
    conf_cols: List[str],
    auc_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    corr_wrong_df: pd.DataFrame,
) -> Dict:
    summary = {
        "n_rows": int(len(df)),
        "n_correct": int(df["is_correct"].sum()) if "is_correct" in df.columns else None,
        "accuracy": float(df["is_correct"].mean()) if "is_correct" in df.columns else None,
        "confidence_columns_used": conf_cols,
        "best_auc_metric": None,
        "best_calibrated_metric": None,
        "best_wrong_error_ranker_abs": None,
    }

    if len(auc_df) > 0:
        best_auc = auc_df.sort_values("AUC", ascending=False).iloc[0].to_dict()
        summary["best_auc_metric"] = best_auc

    calib_real = calib_df.loc[calib_df["is_probability_like"] == True] if len(calib_df) > 0 else pd.DataFrame()
    if len(calib_real) > 0:
        best_calib = calib_real.sort_values("brier", ascending=True).iloc[0].to_dict()
        summary["best_calibrated_metric"] = best_calib

    if len(corr_wrong_df) > 0:
        tmp = corr_wrong_df.loc[corr_wrong_df["error_col"] == "log1p_abs_error"].copy()
        if len(tmp) > 0:
            tmp["abs_spearman"] = tmp["spearman_wrong_only"].abs()
            # pour une "bonne" confiance, on espère souvent une corrélation négative avec l'erreur
            best = tmp.sort_values("spearman_wrong_only", ascending=True).iloc[0].to_dict()
            summary["best_wrong_error_ranker_abs"] = best

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Chemin du fichier .jsonl")
    parser.add_argument("--outdir", default="analysis_results", help="Dossier de sortie")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figs_dir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    raw = load_jsonl(Path(args.jsonl))
    flat = [flatten_record(r) for r in raw]
    df = pd.DataFrame(flat)
    df = to_numeric_safe(df)
    df = build_derived_columns(df)

    conf_cols = infer_confidence_columns(df)
    error_cols = [c for c in ["abs_error", "rel_error", "log_error", "log1p_abs_error"] if c in df.columns]

    # Sauvegarde des données aplaties
    df.to_csv(outdir / "data_flattened.csv", index=False)

    # Descriptif
    desc_cols = list(dict.fromkeys(conf_cols + error_cols + ["is_correct"]))
    desc_df = descriptive_stats(df, desc_cols)
    desc_df.to_csv(outdir / "descriptive_stats.csv", index=False)

    score_summary_rows = []
    for c in conf_cols:
        score_summary_rows.append({
            "confidence_col": c,
            "n_non_null": int(df[c].notna().sum()),
            "mean": float(pd.to_numeric(df[c], errors="coerce").mean()),
            "min": float(pd.to_numeric(df[c], errors="coerce").min()),
            "max": float(pd.to_numeric(df[c], errors="coerce").max()),
            "probability_like_0_1": is_probability_like_series(df[c]),
        })
    score_summary_df = pd.DataFrame(score_summary_rows)
    score_summary_df.to_csv(outdir / "score_summary.csv", index=False)

    # Corrélations
    corr_all_df, corr_wrong_df = correlation_tables(df, conf_cols, error_cols)
    corr_all_df.to_csv(outdir / "corr_all.csv", index=False)
    corr_wrong_df.to_csv(outdir / "corr_wrong_only.csv", index=False)

    # AUC / calibration
    auc_df = auc_analysis(df, conf_cols, seed=args.seed)
    auc_df.to_csv(outdir / "auc_table.csv", index=False)

    calib_df = calibration_analysis(df, conf_cols)
    calib_df.to_csv(outdir / "calibration_table.csv", index=False)

    # Régressions
    logit_uni_df = logistic_univariate(df, conf_cols)
    logit_uni_df.to_csv(outdir / "logistic_univariate.csv", index=False)

    logit_multi_df = logistic_multivariate(df, conf_cols, seed=args.seed)
    logit_multi_df.to_csv(outdir / "logistic_multivariate.csv", index=False)

    linear_wrong_df = linear_wrong_only(df, conf_cols, seed=args.seed)
    linear_wrong_df.to_csv(outdir / "linear_wrong_only.csv", index=False)

    # Figures
    for c in conf_cols:
        save_histogram(df[c], f"Distribution de {c}", figs_dir / f"hist_{c.replace('.', '_')}.png")
        save_boxplot_by_correct(df, c, figs_dir / f"box_by_correct_{c.replace('.', '_')}.png")
        save_scatter_wrong(df, c, "log1p_abs_error", figs_dir / f"scatter_wrong_{c.replace('.', '_')}.png")

        rel_tab = reliability_table(df, c, n_bins=10)
        if len(rel_tab) > 0:
            rel_tab.to_csv(outdir / f"reliability_{c.replace('.', '_')}.csv", index=False)
            save_reliability_plot(
                rel_tab,
                title=f"Reliability plot - {c}",
                outpath=figs_dir / f"reliability_{c.replace('.', '_')}.png",
            )

    # Résumé JSON
    summary = build_summary_json(df, conf_cols, auc_df, calib_df, corr_wrong_df)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console summary
    print("=" * 80)
    print("Analyse terminée")
    print("=" * 80)
    print(f"Fichier lu       : {args.jsonl}")
    print(f"Nombre de lignes : {len(df)}")
    if "is_correct" in df.columns:
        print(f"Accuracy globale : {df['is_correct'].mean():.4f}")
    print(f"Scores étudiés   : {len(conf_cols)}")
    for c in conf_cols:
        print(f"  - {c}")

    print("\nFichiers produits dans :", outdir.resolve())

    if len(auc_df) > 0:
        print("\nTop AUC :")
        print(auc_df.head(10).to_string(index=False))

    if len(calib_df) > 0:
        print("\nCalibration :")
        print(calib_df.head(10).to_string(index=False))

    if len(corr_wrong_df) > 0:
        tmp = corr_wrong_df.loc[corr_wrong_df["error_col"] == "log1p_abs_error"].copy()
        if len(tmp) > 0:
            tmp = tmp.sort_values("spearman_wrong_only", ascending=True)
            print("\nMeilleurs score(s) pour classer la gravité des erreurs (sur les faux) :")
            print(tmp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()