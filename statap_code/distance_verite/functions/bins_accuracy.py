#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def flatten(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, dict) and k in ("prob_all_tokens", "prob_tokens_only"):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out

def wilson_ci(k, n, alpha=0.05):
    """IC Wilson 95% pour une proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    from math import sqrt
    z = 1.959963984540054  # ~ N(0,1) quantile 97.5%
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = (z * sqrt((phat*(1-phat) + z**2/(4*n)) / n)) / denom
    return (center - half, center + half)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--conf_col", default="prob_tokens_only.prob_joint")
    ap.add_argument("--bins", default="0,0.5,0.7,0.8,0.9,0.95,0.97,0.99,1.0000001",
                    help="Bornes séparées par des virgules. Mettre >1 à la fin pour inclure 1.0.")
    ap.add_argument("--out", default="bins_accuracy.csv")
    args = ap.parse_args()

    raw = load_jsonl(Path(args.jsonl))
    df = pd.DataFrame([flatten(r) for r in raw])

    if "abs_error" not in df.columns:
        df["abs_error"] = (pd.to_numeric(df["answer_true"]) - pd.to_numeric(df["answer_model"])).abs()

    df["is_correct"] = (df["abs_error"] == 0).astype(int)

    if args.conf_col not in df.columns:
        raise ValueError(f"Colonne '{args.conf_col}' introuvable.")

    p = pd.to_numeric(df[args.conf_col], errors="coerce")
    y = df["is_correct"]

    data = pd.DataFrame({"p": p, "y": y}).dropna()
    edges = [float(x) for x in args.bins.split(",")]

    # Tranches [a,b)
    data["bin"] = pd.cut(data["p"], bins=edges, right=False, include_lowest=True)

    rows = []
    for b, g in data.groupby("bin", dropna=True):
        n = len(g)
        k = int(g["y"].sum())
        acc = k / n
        lo, hi = wilson_ci(k, n)

        rows.append({
            "bin": str(b),
            "n": n,
            "n_correct": k,
            "n_wrong": n - k,
            "accuracy_%": 100 * acc,
            "wrong_%": 100 * (1 - acc),
            "acc_CI95_lo_%": 100 * lo,
            "acc_CI95_hi_%": 100 * hi,
            "mean_conf_in_bin": float(g["p"].mean()),
        })

    out = pd.DataFrame(rows).sort_values("bin")
    out.to_csv(args.out, index=False)

    # Affichage console lisible
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
    print(f"\nSaved: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()