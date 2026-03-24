#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, norm


# ---------- IO / flatten ----------

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON invalide ligne {i}: {e}") from e
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


# ---------- Stats helpers ----------

def bootstrap_ci(x, stat_fn, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(x[idx]))
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1 - alpha/2)
    return float(lo), float(hi)

def bootstrap_ci_xy(x, y, stat_fn, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(x[idx], y[idx]))
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1 - alpha/2)
    return float(lo), float(hi)

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))


# ---------- Core analyses ----------

def auc_with_test(y, p, seed=0):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    auc = roc_auc_score(y, p)

    # Bootstrap CI
    def stat_fn(idx_auc):
        # idx_auc not used: wrapper via bootstrap_ci_xy
        return 0.0

    lo, hi = bootstrap_ci_xy(p, y, lambda pp, yy: roc_auc_score(yy, pp), seed=seed)

    # “Test” vs 0.5 via bootstrap distribution (two-sided)
    # Compute bootstrap AUCs and p-value
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        aucs.append(roc_auc_score(y[idx], p[idx]))
    aucs = np.array(aucs)
    pval = 2 * min(np.mean(aucs <= 0.5), np.mean(aucs >= 0.5))

    return auc, (lo, hi), float(pval)

def brier_with_ci(y, p, seed=0):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    bs = brier_score_loss(y, p)
    lo, hi = bootstrap_ci_xy(p, y, lambda pp, yy: brier_score_loss(yy, pp), seed=seed)
    return bs, (lo, hi)

def calibration_slope_intercept(y, p):
    """
    Fit: y ~ a + b * logit(p)
    Perfect: a=0, b=1
    """
    y = np.asarray(y).astype(int)
    z = logit(np.asarray(p).astype(float)).reshape(-1, 1)

    # Unpenalized logistic regression (approx; sklearn uses L2 by default => set huge C)
    clf = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    clf.fit(z, y)
    b = float(clf.coef_.ravel()[0])
    a = float(clf.intercept_[0])
    return a, b

def spiegelhalter_z(y, p):
    """
    Spiegelhalter z-test for calibration (global), commonly used with Brier-type residuals.
    z ~ N(0,1) under H0 (approx).
    """
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    e = y - p
    v = p * (1 - p)
    num = np.sum(e * (1 - 2*p))
    den = np.sqrt(np.sum(v * (1 - 2*p)**2))
    if den == 0:
        return np.nan, np.nan
    z = num / den
    pval = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(pval)

def spearman_wrong(log_err, conf, seed=0):
    rho, pval = spearmanr(conf, log_err)
    lo, hi = bootstrap_ci_xy(conf, log_err, lambda c,e: spearmanr(c, e)[0], seed=seed)
    return float(rho), (lo, hi), float(pval)

def reliability_bins(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    b = np.digitize(p, bins) - 1
    out = []
    for k in range(n_bins):
        mask = b == k
        if mask.sum() == 0:
            continue
        out.append({
            "bin": k,
            "n": int(mask.sum()),
            "p_mean": float(p[mask].mean()),
            "acc": float(y[mask].mean()),
        })
    return pd.DataFrame(out)


# ---------- Plots ----------

def plot_reliability(df_bins, outpath):
    plt.figure()
    plt.plot(df_bins["p_mean"], df_bins["acc"], marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability diagram")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_error_vs_conf(conf, log_err, outpath):
    plt.figure()
    plt.scatter(conf, log_err, s=8, alpha=0.25)
    plt.xlabel("Confidence")
    plt.ylabel("log1p(abs_error)  (wrong only)")
    plt.title("Error magnitude vs confidence (wrong only)")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--outdir", default="report_out")
    ap.add_argument("--conf_col", default="prob_tokens_only.prob_joint",
                    help="Colonne de confiance à tester (ex: confidence_verbalized, judge_confidence, prob_tokens_only.prob_joint)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_jsonl(Path(args.jsonl))
    df = pd.DataFrame([flatten_record(r) for r in raw])
    df = to_numeric_safe(df)

    # Derived outcomes
    if "abs_error" not in df.columns and {"answer_true", "answer_model"}.issubset(df.columns):
        df["abs_error"] = (df["answer_true"] - df["answer_model"]).abs()

    df["is_correct"] = (df["abs_error"] == 0).astype(int)
    df["log_abs_error"] = np.log1p(df["abs_error"].astype(float))

    conf_col = args.conf_col
    if conf_col not in df.columns:
        raise ValueError(f"conf_col '{conf_col}' introuvable. Colonnes dispo: {list(df.columns)[:30]} ...")

    # Keep valid
    data = df[[conf_col, "is_correct", "abs_error", "log_abs_error"]].dropna()
    p = data[conf_col].astype(float).values
    y = data["is_correct"].astype(int).values

    # --- Q1: discrimination ---
    auc, (auc_lo, auc_hi), auc_p = auc_with_test(y, p, seed=args.seed)
    bs, (bs_lo, bs_hi) = brier_with_ci(y, p, seed=args.seed)

    # --- Q3: calibration ---
    a, b = calibration_slope_intercept(y, p)
    z, z_p = spiegelhalter_z(y, p)

    # --- Q2: magnitude given wrong ---
    wrong = data[data["is_correct"] == 0]
    rho, (rho_lo, rho_hi), rho_p = spearman_wrong(
        wrong["log_abs_error"].values,
        wrong[conf_col].astype(float).values,
        seed=args.seed
    )

    # Reliability bins + plot
    bins = reliability_bins(y, p, n_bins=10)
    plot_reliability(bins, outdir / f"reliability_{conf_col.replace('.','_')}.png")

    # Error vs conf plot (wrong only)
    if len(wrong) > 0:
        plot_error_vs_conf(
            wrong[conf_col].astype(float).values,
            wrong["log_abs_error"].values,
            outdir / f"error_vs_conf_wrong_{conf_col.replace('.','_')}.png"
        )

    # Save tables
    bins.to_csv(outdir / f"bins_{conf_col.replace('.','_')}.csv", index=False)

    results = pd.DataFrame([{
        "conf_col": conf_col,
        "n": int(len(data)),
        "accuracy": float(y.mean()),
        "mean_conf": float(np.mean(p)),
        "AUC": auc, "AUC_CI_lo": auc_lo, "AUC_CI_hi": auc_hi, "AUC_p": auc_p,
        "Brier": bs, "Brier_CI_lo": bs_lo, "Brier_CI_hi": bs_hi,
        "calib_intercept": a,
        "calib_slope": b,
        "Spiegelhalter_z": z, "Spiegelhalter_p": z_p,
        "Spearman_rho_wrong_logerr": rho,
        "rho_CI_lo": rho_lo, "rho_CI_hi": rho_hi, "rho_p": rho_p,
        "n_wrong": int(len(wrong)),
    }])
    results.to_csv(outdir / f"summary_{conf_col.replace('.','_')}.csv", index=False)

    print(results.to_string(index=False))
    print(f"\nSaved report to: {outdir.resolve()}")
    print("Files: summary_*.csv, bins_*.csv, reliability_*.png, error_vs_conf_wrong_*.png")


if __name__ == "__main__":
    main()