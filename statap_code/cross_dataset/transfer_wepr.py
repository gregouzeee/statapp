"""
Transfer evaluation — apply WEPR trained on TriviaQA to GSM8K (and back)
=======================================================================

Loads two trained WEPR models:
    - WEPR/data/wepr_model.json                  (trained on TriviaQA, §5)
    - cross_dataset/data/wepr_gsm8k_model.json   (trained on GSM8K)

Then computes ROC-AUC of each model when scored on the OTHER dataset.
This tests whether the entropic-feature decision boundary learned on one
task generalises to a different task (open-domain factual QA vs
multi-step arithmetic reasoning).

Output: cross_dataset/data/transfer_results.json with a 2×2 AUC table.

Notes
-----
* Both models share the same WEPR feature layout (K=10, 2K features).
* The TriviaQA model is applied to GSM8K's "full" sub-sequence (whole
  generation), since TriviaQA generations are single-sentence whole
  answers — that's the most apples-to-apples comparison.
* In-domain CV AUCs are reported alongside transfer AUCs so the gap
  is easy to read.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

WEPR_DIR = Path(__file__).resolve().parent.parent / "WEPR"
sys.path.insert(0, str(WEPR_DIR))
from wepr import wepr_features  # noqa: E402


ROOT = Path(__file__).resolve().parent
TRIVIAQA_MODEL = WEPR_DIR / "data" / "wepr_model.json"
TRIVIAQA_JUDGED = WEPR_DIR / "data" / "triviaqa_judged.jsonl"

GSM8K_MODEL = ROOT / "data" / "wepr_gsm8k_model.json"
GSM8K_JUDGED = ROOT / "data" / "gsm8k_judged.jsonl"

OUT_JSON = ROOT / "data" / "transfer_results.json"


def load_judged(path: Path, field: str = "token_data") -> List[Dict]:
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
            td = obj.get(field) or []
            if not td or not td[0].get("top_k"):
                continue
            out.append(obj)
    return out


def build_xy(data: List[Dict], K: int, field: str
             ) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for item in data:
        td = item.get(field) or []
        if not td or not td[0].get("top_k"):
            continue
        feats = wepr_features(td, K)
        X.append(feats["feature_vector"])
        y.append(1 if item["judge_correct"] else 0)
    return np.array(X), np.array(y)


def score_with_model(X: np.ndarray, model: Dict) -> np.ndarray:
    """
    Re-apply a saved WEPR model: standardize with stored mean/std, then
    apply the linear combination beta + gamma.

    The saved model dict has:
        scaler_mean, scaler_std (of length 2K),
        beta (length K+1), gamma (length K).
    """
    K = model["K"]
    mean = np.array(model["scaler_mean"])
    std = np.array(model["scaler_std"])
    beta = np.array(model["beta"])     # length K+1
    gamma = np.array(model["gamma"])   # length K

    # Standardize
    Xs = (X - mean) / np.where(std == 0, 1.0, std)
    # Logistic regression linear combination = intercept + coef·x
    # In our layout: coef[:K] = beta[1:], coef[K:] = gamma
    coef = np.concatenate([beta[1:], gamma])
    intercept = beta[0]
    z = intercept + Xs @ coef
    # Sigmoid -> probability of correct
    return 1.0 / (1.0 + np.exp(-z))


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triviaqa_model", type=str, default=str(TRIVIAQA_MODEL))
    parser.add_argument("--gsm8k_model", type=str, default=str(GSM8K_MODEL))
    parser.add_argument("--triviaqa_data", type=str, default=str(TRIVIAQA_JUDGED))
    parser.add_argument("--gsm8k_data", type=str, default=str(GSM8K_JUDGED))
    parser.add_argument("--gsm8k_field", type=str, default="token_data",
                        choices=["token_data", "reasoning_token_data",
                                 "answer_token_data"])
    parser.add_argument("--output", type=str, default=str(OUT_JSON))
    args = parser.parse_args()

    # Load models
    with open(args.triviaqa_model, "r", encoding="utf-8") as f:
        m_tqa = json.load(f)
    with open(args.gsm8k_model, "r", encoding="utf-8") as f:
        m_gsm_full = json.load(f)
        # Use the "full" subseq model from the GSM8K training output
        if "subseq" in m_gsm_full and "full" in m_gsm_full["subseq"]:
            m_gsm = m_gsm_full["subseq"]["full"]
            if "K" not in m_gsm:
                m_gsm["K"] = m_gsm_full.get("K", 10)
        else:
            m_gsm = m_gsm_full
    K_tqa = m_tqa["K"]
    K_gsm = m_gsm["K"]
    if K_tqa != K_gsm:
        print(f"WARNING: K differs (TriviaQA={K_tqa} vs GSM8K={K_gsm}). "
              "Transfer requires same K.")
    K = K_tqa

    # Load datasets and build X, y
    print("Loading TriviaQA judged data...")
    tqa_data = load_judged(Path(args.triviaqa_data))
    X_tqa, y_tqa = build_xy(tqa_data, K, "token_data")
    print(f"  TriviaQA: {len(y_tqa)} samples (acc={y_tqa.mean():.3f})")

    print("Loading GSM8K judged data...")
    gsm_data = load_judged(Path(args.gsm8k_data))
    X_gsm, y_gsm = build_xy(gsm_data, K, args.gsm8k_field)
    print(f"  GSM8K:    {len(y_gsm)} samples (acc={y_gsm.mean():.3f})")

    # 2x2 AUC table
    p_tqa_on_tqa = score_with_model(X_tqa, m_tqa)
    p_tqa_on_gsm = score_with_model(X_gsm, m_tqa)
    p_gsm_on_gsm = score_with_model(X_gsm, m_gsm)
    p_gsm_on_tqa = score_with_model(X_tqa, m_gsm)

    auc_table = {
        "TriviaQA_model_on_TriviaQA": safe_auc(y_tqa, p_tqa_on_tqa),
        "TriviaQA_model_on_GSM8K":    safe_auc(y_gsm, p_tqa_on_gsm),
        "GSM8K_model_on_GSM8K":       safe_auc(y_gsm, p_gsm_on_gsm),
        "GSM8K_model_on_TriviaQA":    safe_auc(y_tqa, p_gsm_on_tqa),
    }

    print("\n=== AUC transfer table ===")
    for k, v in auc_table.items():
        print(f"  {k:32s}  {v:.4f}")

    out = {
        "K": K,
        "n_triviaqa": int(len(y_tqa)),
        "n_gsm8k": int(len(y_gsm)),
        "gsm8k_field": args.gsm8k_field,
        "auc_in_domain": {
            "TriviaQA": auc_table["TriviaQA_model_on_TriviaQA"],
            "GSM8K":    auc_table["GSM8K_model_on_GSM8K"],
        },
        "auc_transfer": {
            "TriviaQA_to_GSM8K": auc_table["TriviaQA_model_on_GSM8K"],
            "GSM8K_to_TriviaQA": auc_table["GSM8K_model_on_TriviaQA"],
        },
        "auc_full_table": auc_table,
        "report_baseline_section_4_auc": 0.729,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
