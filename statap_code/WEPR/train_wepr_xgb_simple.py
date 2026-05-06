"""
Train WEPR — XGBoost on Entropic Features (version simple)
===========================================================
Variante minimaliste de train_wepr.py : XGBoost à la place de la régression
logistique, avec deux outils d'interprétation :

    1. clf.feature_importances_ (= gain natif de XGBoost)
    2. Les valeurs de SHAP (TreeExplainer)

Lit  : data/triviaqa_judged.jsonl
Écrit: data/wepr_xgb_simple_model.json
       data/wepr_xgb_simple_model.ubj
       figures/xgb_simple_*.png

Usage:
    python train_wepr_xgb_simple.py [--K 10] [--n_splits 5]
"""

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import shap

from wepr import wepr_features


# ──────────────────────── Logging utils ────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@contextmanager
def timed(label: str):
    log(f"▶ {label}…")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log(f"✔ {label} ({time.perf_counter()-t0:.1f}s)")


# ──────────────────────────── Config ────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
JUDGED_JSONL = DATA_DIR / "triviaqa_judged.jsonl"
MODEL_JSON = DATA_DIR / "wepr_xgb_simple_model.json"
MODEL_BIN = DATA_DIR / "wepr_xgb_simple_model.ubj"


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
            td = obj["token_data"]
            if not td or "top_k" not in td[0]:
                continue
            results.append(obj)
    return results


# ──────────────────────── Build features ────────────────────────

def build_dataset(data: List[Dict], K: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    features[0:K]  = mean_by_rank
    features[K:2K] = max_by_rank
    Labels : 1 = correct, 0 = hallucinated.
    """
    X_list, y_list = [], []
    for item in data:
        f = wepr_features(item["token_data"], K)
        X_list.append(f["feature_vector"])
        y_list.append(1 if item["judge_correct"] else 0)

    X = np.array(X_list)
    y = np.array(y_list)
    feature_names = (
        [f"mean_rank_{k+1}" for k in range(K)] +
        [f"max_rank_{k+1}"  for k in range(K)]
    )
    return X, y, feature_names


# ──────────────────────── Train + CV ────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray, params: Dict,
                   n_splits: int, seed: int,
                   verbose_every: int = 50) -> Tuple[float, float, List[float]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        log(f"── Fold {fold+1}/{n_splits} ── train={len(tr)}  test={len(te)}")
        t0 = time.perf_counter()
        clf = xgb.XGBClassifier(**params)
        clf.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=verbose_every)
        prob = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], prob)
        aucs.append(auc)
        log(f"  Fold {fold+1}/{n_splits}: ROC-AUC = {auc:.4f}  ({time.perf_counter()-t0:.1f}s)")
    return float(np.mean(aucs)), float(np.std(aucs)), [float(a) for a in aucs]


# ──────────────────────── Interprétation ────────────────────────

def feature_importance_plot(clf: xgb.XGBClassifier, feature_names: List[str],
                            fig_dir: Path) -> Dict[str, float]:
    """
    clf.feature_importances_ correspond au 'gain' normalisé : pour chaque
    feature, somme des gains apportés par ses splits, puis normalisation
    pour que le tout somme à 1.
    """
    importances = {name: float(v) for name, v in zip(feature_names, clf.feature_importances_)}

    order = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [n for n, _ in order]
    values = [v for _, v in order]

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 0.35 * len(names) + 1))
    plt.barh(names[::-1], values[::-1])
    plt.xlabel("Feature importance (gain normalisé)")
    plt.title("XGBoost — feature_importances_")
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_simple_feature_importance.png", dpi=150)
    plt.close()
    return importances


def shap_analysis(clf: xgb.XGBClassifier, X: np.ndarray, feature_names: List[str],
                  fig_dir: Path) -> Dict:
    """
    SHAP via TreeExplainer (exact pour les modèles d'arbres).
    Identité fondamentale, pour chaque échantillon i :
        log_odds(p_i) = expected_value + sum_j shap_values[i, j]
    """
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    expected_value = float(explainer.expected_value
                           if np.isscalar(explainer.expected_value)
                           else explainer.expected_value[0])

    mean_abs = np.abs(shap_values).mean(axis=0)
    signed_mean = shap_values.mean(axis=0)
    global_importance = {n: float(mean_abs[i])    for i, n in enumerate(feature_names)}
    signed_importance = {n: float(signed_mean[i]) for i, n in enumerate(feature_names)}

    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_simple_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_simple_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "expected_value": expected_value,
        "global_mean_abs": global_importance,
        "signed_mean":     signed_importance,
    }


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train WEPR — XGBoost (version simple)")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--input", type=str, default=str(JUDGED_JSONL))
    parser.add_argument("--output", type=str, default=str(MODEL_JSON))
    args = parser.parse_args()

    with timed("Loading judged data"):
        data = load_judged_data(Path(args.input))
    log(f"Loaded {len(data)} judged samples with top-K logprobs")
    n_correct = sum(1 for d in data if d["judge_correct"])
    n_hallu = len(data) - n_correct
    log(f"  Correct: {n_correct} ({n_correct/len(data):.1%})")
    log(f"  Hallucinated: {n_hallu} ({n_hallu/len(data):.1%})")
    if n_hallu < 5 or n_correct < 5:
        log("ERROR: Not enough samples in each class to train.")
        return

    with timed(f"Building features (K={args.K})"):
        X, y, feature_names = build_dataset(data, args.K)
    log(f"  Feature matrix: {X.shape}")

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=pos_weight,
        random_state=args.seed,
        n_jobs=-1,
    )

    log(f"Cross-validating XGBoost ({args.n_splits} folds, "
        f"n_estimators={args.n_estimators}, max_depth={args.max_depth})…")
    mean_auc, std_auc, fold_aucs = cross_validate(X, y, params, args.n_splits, args.seed)
    log(f"  Mean ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    epr_auc = float(roc_auc_score(y, X[:, :args.K].mean(axis=1)))
    log(f"  EPR baseline ROC-AUC (full data): {epr_auc:.4f}")

    with timed("Training final model on all data"):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X, y, eval_set=[(X, y)], verbose=50)

    with timed("Computing feature importances + plot"):
        importances = feature_importance_plot(clf, feature_names, FIG_DIR)

    with timed("SHAP values + plots (TreeExplainer)"):
        shap_info = shap_analysis(clf, X, feature_names, FIG_DIR)

    print("\nTop features par feature_importances_ :")
    for name, val in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {name:18s}  {val:.4f}")

    print("\nTop features par |SHAP| moyen :")
    top = sorted(shap_info["global_mean_abs"].items(), key=lambda kv: kv[1], reverse=True)[:10]
    for name, val in top:
        signed = shap_info["signed_mean"][name]
        direction = "→ correct" if signed > 0 else "→ hallucination"
        print(f"  {name:18s}  |shap|={val:.4f}   signed={signed:+.4f}  {direction}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(MODEL_BIN))

    result = {
        "K": args.K,
        "feature_names": feature_names,
        "params": params,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "cv_auc_folds": fold_aucs,
        "epr_baseline_auc": epr_auc,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_hallucinated": int(len(y) - y.sum()),
        "feature_importances": importances,
        "shap": shap_info,
        "model_file": str(MODEL_BIN.name),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"Model saved to {MODEL_BIN}")
    log(f"Metrics + importances saved to {output_path}")
    log(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
