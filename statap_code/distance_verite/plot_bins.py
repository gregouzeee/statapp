#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def flatten(r):
    out = {}
    for k,v in r.items():
        if isinstance(v, dict) and k in ("prob_all_tokens","prob_tokens_only"):
            for kk,vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = k/n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    half = z * np.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return center-half, center+half

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--conf_col", default="prob_tokens_only.prob_joint")
    ap.add_argument("--bins", default="0,0.5,0.7,0.8,0.9,0.95,0.97,0.99,1.0000001")
    ap.add_argument("--out", default="reliability_plot.png")
    args = ap.parse_args()

    raw = load_jsonl(args.jsonl)
    df = pd.DataFrame([flatten(r) for r in raw])

    df["abs_error"] = (df["answer_true"] - df["answer_model"]).abs()
    df["is_correct"] = (df["abs_error"] == 0).astype(int)

    p = pd.to_numeric(df[args.conf_col], errors="coerce")
    y = df["is_correct"]

    data = pd.DataFrame({"p":p, "y":y}).dropna()

    edges = [float(x) for x in args.bins.split(",")]
    data["bin"] = pd.cut(data["p"], bins=edges, right=False, include_lowest=True)

    xs, ys, yerr_low, yerr_high, sizes = [], [], [], [], []

    for b, g in data.groupby("bin"):
        n = len(g)
        if n == 0:
            continue
        k = g["y"].sum()
        acc = k/n
        lo, hi = wilson_ci(k,n)

        xs.append(g["p"].mean())
        ys.append(acc)
        yerr_low.append(acc - lo)
        yerr_high.append(hi - acc)
        sizes.append(n)

    xs = np.array(xs)
    ys = np.array(ys)

    plt.figure(figsize=(7,6))
    plt.errorbar(xs, ys, 
                 yerr=[yerr_low, yerr_high], 
                 fmt='o', capsize=4)

    # ligne parfaite calibration
    plt.plot([0,1],[0,1], linestyle='--')

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability diagram\n{args.conf_col}")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid(alpha=0.3)

    plt.savefig(args.out, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()