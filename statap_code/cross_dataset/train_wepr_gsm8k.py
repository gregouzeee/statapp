"""
WEPR on GSM8K — Train and evaluate the WEPR detector on three sub-sequences
==========================================================================

Trains the WEPR logistic-regression model on GSM8K Chain-of-Thought
generations, separately on:

    1) "full"      : all generated tokens (reasoning + final answer)
    2) "reasoning" : only the reasoning sub-sequence
    3) "answer"    : only the final numeric answer sub-sequence

For each sub-sequence we report:
    - cross-validated ROC-AUC (5 folds, stratified)
    - the EPR baseline AUC (unsupervised)
    - the trained beta/gamma coefficients

This mirrors the analysis of §4 of the report (which trained a 64-feature
logistic regression on GSM8K CoT) but with the much more parsimonious
2K+1 = 21-feature WEPR model. It gives a direct answer to the question
"what does WEPR look like on GSM8K, and where does the signal sit?".

Input : data/gsm8k_judged.jsonl
Output: data/wepr_gsm8k_model.json   — coefficients + cv metrics for each subseq
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Reuse WEPR feature extractor from the existing module
WEPR_DIR = Path(__file__).resolve().parent.parent / "WEPR"
sys.path.insert(0, str(WEPR_DIR))
from wepr import wepr_features  # noqa: E402


ROOT = Path(__file__).resolve().parent
IN_JSONL = ROOT / "data" / "gsm8k_judged.jsonl"
OUT_JSON = ROOT / "data" / "wepr_gsm8k_model.json"

SUBSEQ_FIELDS = {
    "full": "token_data",
    "reasoning": "reasoning_token_data",
    "answer": "answer_token_data",
}


def load_judged(path: Path) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("judge_correct") is None:
                continue
            if "token_data" not in obj or not obj["token_data"]:
                continue
            out.append(obj)
    return out


def build_xy(data: List[Dict], K: int, field: str
             ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, qids = [], [], []
    for item in data:
        td = item.get(field) or []
        if not td:
            continue
        # Skip if first token has no top_k (malformed)
        if not td[0].get("top_k"):
            continue
        feats = wepr_features(td, K)
        X.append(feats["feature_vector"])
        y.append(1 if item["judge_correct"] else 0)
        qids.append(item["question_id"])
    return np.array(X), np.array(y), qids


def train_eval(X: np.ndarray, y: np.ndarray, K: int,
               n_splits: int = 5, seed: int = 42) -> Dict:
    """5-fold stratified CV + final fit."""
    if len(np.unique(y)) < 2:
        return {"error": "single-class labels", "n": int(len(y))}

    n_splits_eff = min(n_splits, int(min(np.sum(y == 0), np.sum(y == 1))))
    if n_splits_eff < 2:
        return {"error": f"too few minority samples ({n_splits_eff})",
                "n": int(len(y))}

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

    auc_scores = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(penalty=None, max_iter=2000,
                                 random_state=seed)
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        try:
            auc = roc_auc_score(y[te], prob)
        except ValueError:
            auc = float("nan")
        auc_scores.append(auc)
        print(f"    fold {fold + 1}: AUC = {auc:.4f}")

    # Final fit
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(penalty=None, max_iter=2000, random_state=seed)
    clf.fit(Xs, y)
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])
    beta = np.zeros(K + 1)
    beta[0] = intercept
    beta[1:] = coef[:K]
    gamma = coef[K:]

    # EPR baseline AUC: simple mean of per-rank means
    epr_score = X[:, :K].mean(axis=1)
    try:
        epr_auc = float(roc_auc_score(y, epr_score))
    except ValueError:
        epr_auc = float("nan")

    return {
        "K": K,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_wrong": int(len(y) - y.sum()),
        "cv_n_splits": n_splits_eff,
        "cv_auc_folds": [float(a) for a in auc_scores],
        "cv_auc_mean": float(np.nanmean(auc_scores)),
        "cv_auc_std": float(np.nanstd(auc_scores)),
        "epr_baseline_auc": epr_auc,
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=str, default=str(IN_JSONL))
    parser.add_argument("--output", type=str, default=str(OUT_JSON))
    args = parser.parse_args()

    data = load_judged(Path(args.input))
    n_corr = sum(1 for d in data if d["judge_correct"])
    n_wrong = len(data) - n_corr
    print(f"Loaded {len(data)} judged GSM8K samples "
          f"(correct={n_corr}, wrong={n_wrong})")

    if n_wrong < 5 or n_corr < 5:
        print("Warning: very few samples in one class — AUC will be unstable.")

    results = {"K": args.K, "n_splits": args.n_splits, "subseq": {}}
    for name, field in SUBSEQ_FIELDS.items():
        print(f"\n=== Sub-sequence: {name}  (field={field}) ===")
        X, y, qids = build_xy(data, args.K, field)
        print(f"  X shape: {X.shape}  y mean: {y.mean():.3f}")
        if len(X) == 0:
            results["subseq"][name] = {"error": "no usable rows"}
            continue
        res = train_eval(X, y, args.K, n_splits=args.n_splits,
                         seed=args.seed)
        if "cv_auc_mean" in res:
            print(f"  Mean CV AUC = {res['cv_auc_mean']:.4f} "
                  f"+/- {res['cv_auc_std']:.4f}")
            print(f"  EPR baseline AUC = {res['epr_baseline_auc']:.4f}")
        results["subseq"][name] = res

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
