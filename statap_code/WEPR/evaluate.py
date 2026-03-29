"""
Evaluate WEPR — standalone and combined with other methods
==========================================================
Compares WEPR with EPR, cosine leave-one-out, and combinations.

Reads:
    - data/triviaqa_judged.jsonl
    - data/wepr_model.json

Usage:
    python evaluate.py [--K 10]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report,
)
from sklearn.preprocessing import StandardScaler

from wepr import (
    epr, wepr_features, wepr_score, wepr_sequence_confidence,
    cosine_leave_one_out, cosine_word_leave_one_out,
    per_token_entropy,
)
from train_wepr import load_judged_data


# ──────────────────────────── Config ────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
JUDGED_JSONL = DATA_DIR / "triviaqa_judged.jsonl"
MODEL_JSON = DATA_DIR / "wepr_model.json"


# ──────────────── Feature extraction ─────────────────

def extract_all_features(data: List[Dict], K: int) -> Dict[str, np.ndarray]:
    """
    Extract all method scores for each sample.
    Returns dict of arrays, each of shape (N,).
    """
    n = len(data)

    features = {
        "epr": np.zeros(n),
        "mean_logprob": np.zeros(n),
        "min_logprob": np.zeros(n),
        "sentence_perplexity": np.zeros(n),
        "cosine_mean_importance": np.zeros(n),
        "cosine_max_importance": np.zeros(n),
        "word_cosine_mean_importance": np.zeros(n),
        "word_cosine_max_importance": np.zeros(n),
    }

    # WEPR features (2*K per sample)
    wepr_feat_list = []

    for i, item in enumerate(data):
        td = item["token_data"]

        # EPR
        features["epr"][i] = epr(td, K)

        # Mean / min logprob
        logprobs = [t["log_prob"] for t in td]
        features["mean_logprob"][i] = np.mean(logprobs) if logprobs else 0.0
        features["min_logprob"][i] = np.min(logprobs) if logprobs else 0.0

        # Perplexity
        avg_lp = np.mean(logprobs) if logprobs else 0.0
        features["sentence_perplexity"][i] = np.exp(-avg_lp)

        # WEPR features
        wf = wepr_features(td, K)
        wepr_feat_list.append(wf["feature_vector"])

        # Cosine leave-one-out (token-level)
        cos_scores = cosine_leave_one_out(td)
        if cos_scores:
            importances = [c["importance"] for c in cos_scores]
            features["cosine_mean_importance"][i] = np.mean(importances)
            features["cosine_max_importance"][i] = np.max(importances)

        # Cosine leave-one-out (word-level)
        word_cos = cosine_word_leave_one_out(td)
        if word_cos:
            word_imps = [w["importance"] for w in word_cos]
            features["word_cosine_mean_importance"][i] = np.mean(word_imps)
            features["word_cosine_max_importance"][i] = np.max(word_imps)

    features["wepr_features"] = np.array(wepr_feat_list)  # shape (N, 2*K)

    return features


# ──────────────── Evaluation ─────────────────

def evaluate_single_score(name: str, scores: np.ndarray, y: np.ndarray,
                          higher_is_correct: bool = True):
    """Evaluate a single scalar score."""
    if not higher_is_correct:
        scores = -scores

    try:
        auc = roc_auc_score(y, scores)
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(y, scores)
    except ValueError:
        ap = float("nan")

    return {"name": name, "ROC-AUC": auc, "PR-AUC": ap}


def evaluate_learned_combination(name: str, X: np.ndarray, y: np.ndarray,
                                  n_splits: int = 5, seed: int = 42):
    """Evaluate a learned combination via cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    aps = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(penalty=None, max_iter=1000, random_state=seed)
        clf.fit(X_train_s, y_train)

        y_prob = clf.predict_proba(X_test_s)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
        aps.append(average_precision_score(y_test, y_prob))

    return {
        "name": name,
        "ROC-AUC": float(np.mean(aucs)),
        "ROC-AUC_std": float(np.std(aucs)),
        "PR-AUC": float(np.mean(aps)),
        "PR-AUC_std": float(np.std(aps)),
    }


# ──────────────── Main ─────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate WEPR and combinations")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--input", type=str, default=str(JUDGED_JSONL))
    args = parser.parse_args()

    input_path = Path(args.input)

    # Load data
    data = load_judged_data(input_path)
    print(f"Loaded {len(data)} judged samples")

    y = np.array([1 if d["judge_correct"] else 0 for d in data])
    n_correct = y.sum()
    n_hallu = len(y) - n_correct
    print(f"  Correct: {n_correct} | Hallucinated: {n_hallu}")

    # Extract features
    print(f"\nExtracting features (K={args.K})...")
    features = extract_all_features(data, args.K)

    # ──── Part 1: Single-score methods ────
    print(f"\n{'='*70}")
    print(f" SINGLE-SCORE METHODS (no training, direct AUC)")
    print(f"{'='*70}")

    single_results = []

    # EPR (higher entropy → more likely hallucinated → higher_is_correct=False)
    single_results.append(evaluate_single_score(
        "EPR (entropy)", features["epr"], y, higher_is_correct=False))

    # Mean logprob (higher → more confident → correct)
    single_results.append(evaluate_single_score(
        "Mean log-prob", features["mean_logprob"], y, higher_is_correct=True))

    # Min logprob
    single_results.append(evaluate_single_score(
        "Min log-prob", features["min_logprob"], y, higher_is_correct=True))

    # Perplexity (lower → more confident)
    single_results.append(evaluate_single_score(
        "Perplexity", features["sentence_perplexity"], y, higher_is_correct=False))

    # Cosine mean importance (higher → tokens more different → uncertain)
    single_results.append(evaluate_single_score(
        "Cosine mean imp.", features["cosine_mean_importance"], y, higher_is_correct=False))

    # Cosine max importance
    single_results.append(evaluate_single_score(
        "Cosine max imp.", features["cosine_max_importance"], y, higher_is_correct=False))

    # Word cosine
    single_results.append(evaluate_single_score(
        "Word cosine mean", features["word_cosine_mean_importance"], y, higher_is_correct=False))
    single_results.append(evaluate_single_score(
        "Word cosine max", features["word_cosine_max_importance"], y, higher_is_correct=False))

    print(f"\n  {'Method':<25s}  {'ROC-AUC':>10s}  {'PR-AUC':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}")
    for r in single_results:
        print(f"  {r['name']:<25s}  {r['ROC-AUC']:>10.4f}  {r['PR-AUC']:>10.4f}")

    # ──── Part 2: Learned methods (cross-validated) ────
    print(f"\n{'='*70}")
    print(f" LEARNED METHODS (5-fold cross-validation)")
    print(f"{'='*70}")

    learned_results = []

    # WEPR alone
    learned_results.append(evaluate_learned_combination(
        "WEPR", features["wepr_features"], y))

    # WEPR + cosine features
    wepr_cosine = np.column_stack([
        features["wepr_features"],
        features["cosine_mean_importance"],
        features["cosine_max_importance"],
    ])
    learned_results.append(evaluate_learned_combination(
        "WEPR + cosine", wepr_cosine, y))

    # WEPR + word cosine features
    wepr_word_cosine = np.column_stack([
        features["wepr_features"],
        features["word_cosine_mean_importance"],
        features["word_cosine_max_importance"],
    ])
    learned_results.append(evaluate_learned_combination(
        "WEPR + word cosine", wepr_word_cosine, y))

    # WEPR + all cosine features
    wepr_all_cosine = np.column_stack([
        features["wepr_features"],
        features["cosine_mean_importance"],
        features["cosine_max_importance"],
        features["word_cosine_mean_importance"],
        features["word_cosine_max_importance"],
    ])
    learned_results.append(evaluate_learned_combination(
        "WEPR + all cosine", wepr_all_cosine, y))

    # Cosine alone (learned)
    cosine_only = np.column_stack([
        features["cosine_mean_importance"],
        features["cosine_max_importance"],
        features["word_cosine_mean_importance"],
        features["word_cosine_max_importance"],
    ])
    learned_results.append(evaluate_learned_combination(
        "Cosine only (learned)", cosine_only, y))

    # All features combined
    all_features = np.column_stack([
        features["wepr_features"],
        features["cosine_mean_importance"],
        features["cosine_max_importance"],
        features["word_cosine_mean_importance"],
        features["word_cosine_max_importance"],
        features["mean_logprob"],
        features["min_logprob"],
    ])
    learned_results.append(evaluate_learned_combination(
        "All features", all_features, y))

    print(f"\n  {'Method':<25s}  {'ROC-AUC':>10s}  {'(std)':>8s}  {'PR-AUC':>10s}  {'(std)':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for r in learned_results:
        print(f"  {r['name']:<25s}  {r['ROC-AUC']:>10.4f}  {r['ROC-AUC_std']:>7.4f}  {r['PR-AUC']:>10.4f}  {r['PR-AUC_std']:>7.4f}")

    # ──── Part 3: Save comparison ────
    output_path = DATA_DIR / "evaluation_results.json"
    results = {
        "K": args.K,
        "n_samples": len(y),
        "single_score": single_results,
        "learned": learned_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
