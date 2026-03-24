#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def flatten(r):
    out = {}
    for k,v in r.items():
        if isinstance(v, dict) and k in ("prob_all_tokens", "prob_tokens_only"):
            for kk,vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def calib_slope_intercept(y, p):
    z = logit(p).reshape(-1,1)
    clf = LogisticRegression(C=1e6, max_iter=1000)
    clf.fit(z, y)
    return float(clf.intercept_[0]), float(clf.coef_[0][0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()

    raw = load_jsonl(args.jsonl)
    df = pd.DataFrame([flatten(r) for r in raw])

    df["abs_error"] = (df["answer_true"] - df["answer_model"]).abs()
    df["is_correct"] = (df["abs_error"] == 0).astype(int)
    df["log_abs_error"] = np.log1p(df["abs_error"])

    conf_cols = [
        "prob_tokens_only.prob_joint",
        "prob_tokens_only.prob_geo_mean",
        "prob_all_tokens.prob_joint",
        "prob_all_tokens.prob_geo_mean",
        "confidence_verbalized",
        "judge_confidence",
    ]

    results = []

    for col in conf_cols:
        if col not in df.columns:
            continue

        data = df[[col, "is_correct", "log_abs_error"]].dropna()
        p = data[col].astype(float).values
        y = data["is_correct"].values

        auc = roc_auc_score(y, p)
        brier = brier_score_loss(y, p)
        intercept, slope = calib_slope_intercept(y, p)

        wrong = data[data["is_correct"]==0]
        if len(wrong)>0:
            rho, _ = spearmanr(wrong[col], wrong["log_abs_error"])
        else:
            rho = np.nan

        results.append({
            "metric": col,
            "AUC": auc,
            "Brier": brier,
            "calib_intercept": intercept,
            "calib_slope": slope,
            "mean_conf": float(np.mean(p)),
            "accuracy": float(np.mean(y)),
            "rho_wrong_logerr": rho
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("AUC", ascending=False)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()