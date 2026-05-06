"""
Train WEPR — XGBoost on Entropic Features
==========================================
Variante non-linéaire de train_wepr.py : on remplace la régression logistique
par un XGBoost et on étudie ce que le modèle apprend via :

    1. Les importances natives de XGBoost (gain, weight, cover)
    2. La permutation importance (sklearn)
    3. Les valeurs de SHAP (TreeExplainer) — importance globale et locale

Lit  : data/triviaqa_judged.jsonl
Écrit: data/wepr_xgb_model.json   (métriques + importances)
       data/wepr_xgb_model.ubj    (modèle XGBoost sérialisé)
       figures/xgb_*.png          (graphiques d'importance et SHAP)

Usage:
    python train_wepr_xgb.py [--K 10] [--n_splits 5]
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

from wepr import wepr_features


# ──────────────────────── Logging utils ────────────────────────

def log(msg: str) -> None:
    """Print horodaté, flush immédiat pour suivi temps réel."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@contextmanager
def timed(label: str):
    log(f"▶ {label}…")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"✔ {label} ({dt:.1f}s)")


# ──────────────────────────── Config ────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
JUDGED_JSONL = DATA_DIR / "triviaqa_judged.jsonl"
MODEL_JSON = DATA_DIR / "wepr_xgb_model.json"
MODEL_BIN = DATA_DIR / "wepr_xgb_model.ubj"


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

def build_dataset(data: List[Dict], K: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Même feature vector que train_wepr.py :
        features[0:K]  = mean_by_rank
        features[K:2K] = max_by_rank
    Labels : 1 = correct, 0 = hallucinated.
    On renvoie aussi la liste des noms de features pour SHAP.
    """
    X_list, y_list, qids = [], [], []
    for item in data:
        features = wepr_features(item["token_data"], K)
        X_list.append(features["feature_vector"])
        y_list.append(1 if item["judge_correct"] else 0)
        qids.append(item["question_id"])

    X = np.array(X_list)
    y = np.array(y_list)

    feature_names = (
        [f"mean_rank_{k+1}" for k in range(K)] +
        [f"max_rank_{k+1}"  for k in range(K)]
    )
    return X, y, qids, feature_names


# ──────────────────────── Train + CV ────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray, params: Dict,
                   n_splits: int = 5, seed: int = 42,
                   verbose_every: int = 50) -> Tuple[float, float, List[float]]:
    """Stratified K-fold CV sur XGBoost. Retourne (mean_auc, std_auc, fold_aucs)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        log(f"── Fold {fold+1}/{n_splits} ── train={len(tr)}  test={len(te)}")
        t0 = time.perf_counter()
        clf = xgb.XGBClassifier(**params)
        # verbose=N → XGBoost imprime l'eval AUC tous les N arbres
        clf.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=verbose_every)
        prob = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], prob)
        aucs.append(auc)
        log(f"  Fold {fold+1}/{n_splits}: ROC-AUC = {auc:.4f}  ({time.perf_counter()-t0:.1f}s)")
    return float(np.mean(aucs)), float(np.std(aucs)), [float(a) for a in aucs]


def fit_final(X: np.ndarray, y: np.ndarray, params: Dict,
              verbose_every: int = 50) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y, eval_set=[(X, y)], verbose=verbose_every)
    return clf


# ──────────────────────── Interprétation ────────────────────────

def builtin_importances(clf: xgb.XGBClassifier, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    XGBoost expose 3 types d'importance :
        - 'weight' : nombre de splits utilisant la feature
        - 'gain'   : gain moyen apporté par les splits sur cette feature (le plus interprétable)
        - 'cover'  : nb moyen d'échantillons concernés par les splits sur cette feature
    """
    booster = clf.get_booster()
    importances = {}
    for kind in ("weight", "gain", "cover"):
        raw = booster.get_score(importance_type=kind)
        # raw indexe par 'f0', 'f1', ... → on remap sur les vrais noms
        named = {feature_names[int(k[1:])]: float(v) for k, v in raw.items()}
        # complète à 0 les features jamais utilisées
        for name in feature_names:
            named.setdefault(name, 0.0)
        importances[kind] = named
    return importances


def permutation_imp(clf: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray,
                    feature_names: List[str], seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Permutation importance : chute de ROC-AUC quand on permute une feature."""
    res = permutation_importance(
        clf, X, y, scoring="roc_auc",
        n_repeats=20, random_state=seed, n_jobs=-1,
    )
    return {
        name: {"mean": float(res.importances_mean[i]),
               "std":  float(res.importances_std[i])}
        for i, name in enumerate(feature_names)
    }


def shap_analysis(clf: xgb.XGBClassifier, X: np.ndarray, feature_names: List[str],
                  fig_dir: Path) -> Dict:
    """
    SHAP via TreeExplainer (exact pour les modèles d'arbres).
    Pour chaque échantillon i et chaque feature j, shap_values[i, j] est la
    contribution (en log-odds) de la feature j à la prédiction de i, par rapport
    à la prédiction moyenne du modèle (la 'expected value' de l'explainer).

    Identité fondamentale :
        log_odds(p_i) = expected_value + sum_j shap_values[i, j]
    """
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)  # (n_samples, n_features)
    expected_value = float(explainer.expected_value
                           if np.isscalar(explainer.expected_value)
                           else explainer.expected_value[0])

    # Importance globale = mean(|shap|) sur tous les échantillons
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_importance = {
        name: float(mean_abs[i]) for i, name in enumerate(feature_names)
    }

    # Effet signé moyen = mean(shap) — indique si la feature pousse plutôt
    # vers la classe 1 (correct) ou la classe 0 (hallucination) en moyenne.
    signed_mean = shap_values.mean(axis=0)
    signed_importance = {
        name: float(signed_mean[i]) for i, name in enumerate(feature_names)
    }

    # ─── Plots ───
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "expected_value": expected_value,
        "global_mean_abs": global_importance,
        "signed_mean":     signed_importance,
    }


def plot_importances(importances: Dict[str, Dict[str, float]],
                     perm_imp: Dict[str, Dict[str, float]],
                     fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Built-in (gain)
    gain = importances["gain"]
    names = list(gain.keys())
    values = [gain[n] for n in names]
    order = np.argsort(values)[::-1]

    plt.figure(figsize=(8, 0.35 * len(names) + 1))
    plt.barh([names[i] for i in order][::-1], [values[i] for i in order][::-1])
    plt.xlabel("XGBoost gain")
    plt.title("Importance des features (gain)")
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_gain.png", dpi=150)
    plt.close()

    # Permutation
    means = np.array([perm_imp[n]["mean"] for n in names])
    stds  = np.array([perm_imp[n]["std"]  for n in names])
    order = np.argsort(means)[::-1]

    plt.figure(figsize=(8, 0.35 * len(names) + 1))
    plt.barh([names[i] for i in order][::-1],
             means[order][::-1], xerr=stds[order][::-1])
    plt.xlabel("Chute moyenne de ROC-AUC")
    plt.title("Permutation importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "xgb_permutation.png", dpi=150)
    plt.close()


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train WEPR with XGBoost + SHAP")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--input", type=str, default=str(JUDGED_JSONL))
    parser.add_argument("--output", type=str, default=str(MODEL_JSON))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # ─── Données ───
    with timed("Loading judged data"):
        data = load_judged_data(input_path)
    log(f"Loaded {len(data)} judged samples with top-K logprobs")
    n_correct = sum(1 for d in data if d["judge_correct"])
    n_hallu = len(data) - n_correct
    log(f"  Correct: {n_correct} ({n_correct/len(data):.1%})")
    log(f"  Hallucinated: {n_hallu} ({n_hallu/len(data):.1%})")

    if n_hallu < 5 or n_correct < 5:
        log("ERROR: Not enough samples in each class to train.")
        return

    with timed(f"Building features (K={args.K})"):
        X, y, qids, feature_names = build_dataset(data, args.K)
    log(f"  Feature matrix: {X.shape}")

    # XGBoost gère bien des features non-standardisées : on garde X tel quel
    # pour préserver l'interprétabilité (les seuils des splits sont lisibles).
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

    # ─── CV ───
    log(f"Cross-validating XGBoost ({args.n_splits} folds, "
        f"n_estimators={args.n_estimators}, max_depth={args.max_depth})…")
    mean_auc, std_auc, fold_aucs = cross_validate(X, y, params, args.n_splits, args.seed)
    log(f"  Mean ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # Baseline EPR
    epr_scores = X[:, :args.K].mean(axis=1)
    epr_auc = float(roc_auc_score(y, epr_scores))
    log(f"  EPR baseline ROC-AUC (full data): {epr_auc:.4f}")

    # ─── Modèle final ───
    with timed("Training final model on all data"):
        clf = fit_final(X, y, params)

    # ─── Interprétation ───
    with timed("Computing built-in importances (weight / gain / cover)"):
        importances = builtin_importances(clf, feature_names)

    with timed(f"Permutation importance (n_repeats=20, n_features={len(feature_names)})"):
        perm = permutation_imp(clf, X, y, feature_names, seed=args.seed)

    with timed("SHAP values + plots (TreeExplainer)"):
        shap_info = shap_analysis(clf, X, feature_names, FIG_DIR)

    with timed("Plotting importance bar charts"):
        plot_importances(importances, perm, FIG_DIR)

    # ─── Affichage ───
    def topk(d: Dict[str, float], k: int = 10):
        return sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]

    print("\nTop features par GAIN (XGBoost) :")
    for name, val in topk(importances["gain"]):
        print(f"  {name:18s}  {val:10.4f}")

    print("\nTop features par PERMUTATION IMPORTANCE :")
    for name, val in sorted(perm.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:10]:
        print(f"  {name:18s}  {val['mean']:+.4f}  (± {val['std']:.4f})")

    print("\nTop features par |SHAP| moyen :")
    for name, val in topk(shap_info["global_mean_abs"]):
        signed = shap_info["signed_mean"][name]
        direction = "→ correct" if signed > 0 else "→ hallucination"
        print(f"  {name:18s}  |shap|={val:.4f}   signed={signed:+.4f}  {direction}")

    # ─── Sauvegarde ───
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
        "importances": importances,
        "permutation_importance": perm,
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
