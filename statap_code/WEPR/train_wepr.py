"""
Train WEPR — Logistic Regression on Entropic Features
======================================================
Trains the WEPR model (beta and gamma coefficients) on the judged
TriviaQA dataset using logistic regression.

Reads: data/triviaqa_judged.jsonl
Saves: data/wepr_model.json (coefficients + metadata)

Usage:
    python train_wepr.py [--K 10] [--test_frac 0.2]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from wepr import entropic_contributions, wepr_features


# ──────────────────────────── Config ────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
JUDGED_JSONL = DATA_DIR / "triviaqa_judged.jsonl"
MODEL_JSON = DATA_DIR / "wepr_model.json"


# ──────────────────────── Load data ────────────────────────

def load_judged_data(path: Path) -> List[Dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("judge_correct") is None:
                continue
            if "token_data" not in obj:
                continue
            # Check that token_data has top_k
            td = obj["token_data"]
            if not td or "top_k" not in td[0]:
                continue
            results.append(obj)
    return results


# ──────────────────────── Build features ────────────────────────

def build_dataset(data: List[Dict], K: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix X and label vector y from judged data.

    Features per sequence:
        - mean_by_rank (K values): mean of s(k,j) per rank
        - max_by_rank (K values): max of s(k,j) per rank
        Total: 2*K features

    Labels: 1 = correct (non-hallucinated), 0 = hallucinated
    """
    X_list = []
    y_list = []
    qids = []

    for item in data:
        features = wepr_features(item["token_data"], K)
        fv = features["feature_vector"]
        label = 1 if item["judge_correct"] else 0

        X_list.append(fv)
        y_list.append(label)
        qids.append(item["question_id"])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, qids


# ──────────────────────── Train ────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray, K: int,
                       n_splits: int = 5, seed: int = 42) -> Dict:
    """
    Train WEPR with stratified k-fold cross-validation.

    The logistic regression learns:
        - beta (K+1 values): intercept + weight per rank for mean contributions
        - gamma (K values): weight per rank for max contributions

    In our feature vector layout:
        features[0:K] = mean_by_rank  -> these get beta[1:K+1]
        features[K:2K] = max_by_rank  -> these get gamma[1:K]
        intercept -> beta[0]

    Returns dict with model coefficients and evaluation metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    auc_scores = []
    all_y_true = []
    all_y_prob = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic regression (no regularization, like the paper)
        clf = LogisticRegression(
            penalty=None,
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_train_scaled, y_train)

        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(f"  Fold {fold+1}/{n_splits}: ROC-AUC = {auc:.4f}")

    # Train final model on all data
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)

    clf_final = LogisticRegression(
        penalty=None,
        max_iter=1000,
        random_state=seed,
    )
    clf_final.fit(X_scaled, y)

    # Extract beta and gamma from the final model
    # Layout: coef[0:K] = weights for mean_by_rank, coef[K:2K] = weights for max_by_rank
    coef = clf_final.coef_[0]
    intercept = clf_final.intercept_[0]

    beta = np.zeros(K + 1)
    beta[0] = intercept
    beta[1:] = coef[:K]

    gamma = coef[K:]

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"\n  Mean ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # Also compute EPR baseline AUC for comparison
    epr_scores = X[:, :K].mean(axis=1)  # mean entropy = EPR
    epr_auc = roc_auc_score(y, epr_scores)
    print(f"  EPR baseline ROC-AUC (on full data): {epr_auc:.4f}")

    return {
        "K": K,
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
        "scaler_mean": scaler_final.mean_.tolist(),
        "scaler_std": scaler_final.scale_.tolist(),
        "cv_auc_mean": float(mean_auc),
        "cv_auc_std": float(std_auc),
        "cv_auc_folds": [float(a) for a in auc_scores],
        "epr_baseline_auc": float(epr_auc),
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_hallucinated": int(len(y) - y.sum()),
    }


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train WEPR model")
    parser.add_argument("--K", type=int, default=10, help="Number of top-K logprobs to use")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=str, default=str(JUDGED_JSONL))
    parser.add_argument("--output", type=str, default=str(MODEL_JSON))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load data
    data = load_judged_data(input_path)
    print(f"Loaded {len(data)} judged samples with top-K logprobs")

    n_correct = sum(1 for d in data if d["judge_correct"])
    n_hallu = len(data) - n_correct
    print(f"  Correct: {n_correct} ({n_correct/len(data):.1%})")
    print(f"  Hallucinated: {n_hallu} ({n_hallu/len(data):.1%})")

    if n_hallu < 5 or n_correct < 5:
        print("ERROR: Not enough samples in each class to train.")
        return

    # Build features
    print(f"\nBuilding features with K={args.K}...")
    X, y, qids = build_dataset(data, args.K)
    print(f"  Feature matrix: {X.shape}")

    # Train
    print(f"\nTraining WEPR (K={args.K}, {args.n_splits}-fold CV)...")
    result = train_and_evaluate(X, y, args.K, n_splits=args.n_splits, seed=args.seed)

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nModel saved to {output_path}")

    # Print coefficients
    print(f"\nLearned coefficients (K={args.K}):")
    print(f"  beta_0 (intercept): {result['beta'][0]:.4f}")
    for k in range(args.K):
        print(f"  beta_{k+1} (rank {k+1} mean): {result['beta'][k+1]:.4f}")
    for k in range(args.K):
        print(f"  gamma_{k+1} (rank {k+1} max):  {result['gamma'][k]:.4f}")


if __name__ == "__main__":
    main()
