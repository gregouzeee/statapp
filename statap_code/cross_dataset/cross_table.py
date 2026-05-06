"""
Cross-dataset comparison table — confidence scores × datasets
=============================================================

Builds a single CSV (and a LaTeX rendering) summarising AUC, ECE and
Brier of every confidence score that is computable on each dataset:

    Datasets:
      - TriviaQA-num    (3 397 questions, §3 of the report)
      - TriviaQA-WEPR   (73 793 questions, §5 of the report)
      - ELI5            (100 questions, long-form, judge in {1..5})
      - GSM8K           (300 questions, generated here, CoT)
      - MMLU            (500 questions, §2 of the report — softmax over A/B/C/D)

    Scores (where applicable):
      - confidence_verbalized           (TriviaQA-num, MMLU-verb if present, GSM8K via
                                         self-reported confidence — when stored)
      - judge_confidence                (TriviaQA-num, ELI5 accuracy/5)
      - prob_joint_seq                  (any dataset with token logprobs)
      - prob_geo_mean_seq               idem
      - perplexity_seq                  idem
      - p_min_seq                       idem
      - epr / wepr                       (datasets with stored top-K logprobs)
      - semantic_entropy                (TriviaQA-num if SE was generated, ELI5)
      - selfcheck                       (ELI5)
      - max_softmax                     (MMLU)

For each (dataset, score) cell that exists, we report:
    n, AUC (correct/incorrect discrimination), ECE, Brier, Spearman
    (when a "distance to truth" target is available — i.e. TriviaQA-num).

Outputs:
    cross_dataset/data/cross_table.csv       (long format)
    cross_dataset/data/cross_table_wide.csv  (pivot AUC)
    cross_dataset/data/cross_table.tex       (LaTeX, paste into the report)
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("cross_table")


ROOT = Path(__file__).resolve().parent
TRIVIAQA_NUM = (ROOT.parent / "distance_verite"
                / "triviaqa_numeric_results.jsonl")
TRIVIAQA_WEPR = ROOT.parent / "WEPR" / "data" / "triviaqa_judged.jsonl"
ELI5_JUDGED = (ROOT.parent / "text_uncertainty" / "eli5_uncertainty"
               / "eli5_judged.jsonl")
ELI5_SE = (ROOT.parent / "text_uncertainty" / "eli5_uncertainty"
           / "eli5_semantic_entropy.jsonl")
ELI5_SC = (ROOT.parent / "text_uncertainty" / "eli5_uncertainty"
           / "eli5_selfcheck.jsonl")
MMLU = (ROOT.parent / "Conformal_prediction"
        / "mmlu_500_confidence_probs.jsonl")
GSM8K_JUDGED = ROOT / "data" / "gsm8k_judged.jsonl"
SE_TRIVIAQA_NUM = ROOT / "data" / "se_triviaqa_numeric.jsonl"

# Newly generated top-K files (to fill EPR/WEPR holes)
TRIVIAQA_NUM_TOPK = ROOT / "data" / "triviaqa_num_topk.jsonl"
ELI5_TOPK = ROOT / "data" / "eli5_topk.jsonl"
SE_GSM8K = ROOT / "data" / "se_gsm8k.jsonl"

OUT_DIR = ROOT / "data"
OUT_LONG = OUT_DIR / "cross_table.csv"
OUT_WIDE = OUT_DIR / "cross_table_wide.csv"
OUT_TEX = OUT_DIR / "cross_table.tex"


# ──────────────────────── Metrics ────────────────────────

def expected_calibration_error(y_true: np.ndarray, p: np.ndarray,
                                n_bins: int = 10) -> float:
    """Standard ECE on equal-width bins."""
    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        acc = float(y_true[mask].mean())
        conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return None


def safe_brier(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(y) == 0:
        return None
    p = np.clip(p, 0.0, 1.0)
    try:
        return float(brier_score_loss(y, p))
    except Exception:
        return None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 5:
        return None
    try:
        rho, _ = spearmanr(x, y)
        return float(rho) if rho is not None and not math.isnan(rho) else None
    except Exception:
        return None


# ──────────────────────── Score extraction helpers ────────────────────────

def seq_scores_from_token_data(token_data: List[Dict]
                                ) -> Dict[str, float]:
    """Return prob_joint, prob_geo_mean, perplexity, p_min from chosen
    log-probs."""
    if not token_data:
        return {}
    lps = [t["log_prob"] for t in token_data if "log_prob" in t]
    if not lps:
        return {}
    n = len(lps)
    s = sum(lps)
    avg = s / n
    return {
        "prob_joint_seq":   math.exp(s),
        "prob_geo_mean_seq": math.exp(avg),
        "perplexity_seq":   math.exp(-avg),
        "p_min_seq":        math.exp(min(lps)),
        "logprob_mean_seq": avg,
        "n_tokens":         n,
    }


def epr_score(token_data: List[Dict], K: int = 10) -> Optional[float]:
    """Lightweight EPR (mean over positions of truncated entropy on top-K)."""
    if not token_data or "top_k" not in token_data[0]:
        return None
    H = []
    for td in token_data:
        topk = td.get("top_k") or []
        h = 0.0
        for c in topk[:K]:
            p = math.exp(c["log_prob"])
            if p > 0:
                h -= p * math.log2(p)
        H.append(h)
    if not H:
        return None
    return float(np.mean(H))


def wepr_with_model(token_data: List[Dict], model: Dict) -> Optional[float]:
    """Apply a saved WEPR model -> sigmoid score in [0,1]."""
    if not token_data or "top_k" not in token_data[0]:
        return None
    K = model["K"]
    # Compute features
    s = np.zeros((len(token_data), K))
    for j, td in enumerate(token_data):
        topk = sorted(td.get("top_k") or [],
                      key=lambda x: x["log_prob"], reverse=True)
        for k in range(min(K, len(topk))):
            p = math.exp(topk[k]["log_prob"])
            if p > 0:
                s[j, k] = -p * math.log2(p)
    if len(s) == 0:
        return None
    fv = np.concatenate([s.mean(axis=0), s.max(axis=0)])
    mean = np.array(model["scaler_mean"])
    std = np.array(model["scaler_std"])
    fvs = (fv - mean) / np.where(std == 0, 1.0, std)
    beta = np.array(model["beta"])
    gamma = np.array(model["gamma"])
    coef = np.concatenate([beta[1:], gamma])
    z = float(beta[0] + fvs @ coef)
    return 1.0 / (1.0 + math.exp(-z))


# ──────────────────────── Per-dataset score loading ────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def scores_triviaqa_num(items: List[Dict],
                        se_lookup: Dict[str, float],
                        topk_lookup: Dict[str, Dict],
                        wepr_model: Optional[Dict]) -> List[Dict]:
    """One row per item with all available scores + correct/error labels.

    `topk_lookup` maps question -> dict with `token_data` (top-K logprobs)
    from the regenerated TriviaQA-num run. When present we add EPR / WEPR
    columns; otherwise they are simply absent for this row.
    """
    rows = []
    for it in items:
        true = it.get("answer_true")
        pred = it.get("answer_model")
        if true is None or pred is None:
            continue
        try:
            true_f = float(true)
            pred_f = float(pred)
        except (TypeError, ValueError):
            continue
        correct = int(math.isclose(pred_f, true_f, rel_tol=1e-3, abs_tol=1e-3))
        abs_err = abs(pred_f - true_f)

        row = {
            "dataset": "TriviaQA-num",
            "correct": correct,
            "abs_error": abs_err,
            "confidence_verbalized": it.get("confidence_verbalized"),
        }
        pall = it.get("prob_all_tokens") or {}
        ptok = it.get("prob_tokens_only") or {}
        for src, prefix in [(pall, "all"), (ptok, "seq")]:
            if not src:
                continue
            if "prob_joint" in src:
                row[f"prob_joint_{prefix}"] = src["prob_joint"]
            if "prob_geo_mean" in src:
                row[f"prob_geo_mean_{prefix}"] = src["prob_geo_mean"]
            if "perplexity" in src:
                row[f"perplexity_{prefix}"] = src["perplexity"]
            if "p_min" in src:
                row[f"p_min_{prefix}"] = src["p_min"]
        q = it.get("question")
        if q in se_lookup:
            row["semantic_entropy"] = se_lookup[q]

        # Fill EPR / WEPR if we have a regenerated row with top-K
        topk_entry = topk_lookup.get(q)
        if topk_entry:
            td = topk_entry.get("token_data") or []
            if td and "top_k" in td[0]:
                ep = epr_score(td)
                if ep is not None:
                    row["epr"] = ep
                if wepr_model is not None:
                    wp = wepr_with_model(td, wepr_model)
                    if wp is not None:
                        row["wepr"] = wp
        rows.append(row)
    return rows


def scores_triviaqa_wepr(items: List[Dict],
                         wepr_model: Optional[Dict]) -> List[Dict]:
    rows = []
    n_total = len(items)
    t0 = time.monotonic()
    for i, it in enumerate(items):
        if i and i % 5000 == 0:
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n_total - i) / rate if rate > 0 else float("inf")
            log.info(f"  TriviaQA-WEPR: scoring {i}/{n_total} "
                     f"({rate:.0f} q/s, ETA {eta:.0f}s)")
        if it.get("judge_correct") is None:
            continue
        td = it.get("token_data") or []
        if not td:
            continue
        seq = seq_scores_from_token_data(td)
        row = {
            "dataset": "TriviaQA-WEPR",
            "correct": int(bool(it["judge_correct"])),
            **seq,
        }
        ep = epr_score(td)
        if ep is not None:
            row["epr"] = ep
        if wepr_model is not None:
            wp = wepr_with_model(td, wepr_model)
            if wp is not None:
                row["wepr"] = wp
        rows.append(row)
    return rows


def scores_eli5(items: List[Dict],
                se_lookup: Dict[str, float],
                sc_lookup: Dict[str, float],
                topk_lookup: Dict[str, Dict],
                wepr_model: Optional[Dict]) -> List[Dict]:
    rows = []
    for it in items:
        js = it.get("judge_scores") or {}
        acc = js.get("accuracy")
        if acc is None:
            continue
        correct = int(acc >= 4)
        row = {
            "dataset": "ELI5",
            "correct": correct,
            "judge_accuracy_5": float(acc),
            "judge_confidence": float(acc) / 5.0,
        }
        if it.get("overall_avg_log_prob") is not None:
            avg = it["overall_avg_log_prob"]
            row["prob_geo_mean_seq"] = math.exp(avg)
            row["perplexity_seq"] = math.exp(-avg)
        if it.get("overall_perplexity") is not None:
            row.setdefault("perplexity_seq", it["overall_perplexity"])
        td = it.get("token_data") or []
        if td:
            seq = seq_scores_from_token_data(td)
            for k, v in seq.items():
                row.setdefault(k, v)
        qid = it.get("question_id")
        if qid in se_lookup:
            row["semantic_entropy"] = se_lookup[qid]
        if qid in sc_lookup:
            row["selfcheck"] = sc_lookup[qid]

        # Fill EPR / WEPR from regenerated top-K data
        topk_entry = topk_lookup.get(qid)
        if topk_entry:
            td2 = topk_entry.get("token_data") or []
            if td2 and "top_k" in td2[0]:
                ep = epr_score(td2)
                if ep is not None:
                    row["epr"] = ep
                if wepr_model is not None:
                    wp = wepr_with_model(td2, wepr_model)
                    if wp is not None:
                        row["wepr"] = wp
        rows.append(row)
    return rows


def scores_gsm8k(items: List[Dict],
                 wepr_model: Optional[Dict] = None,
                 se_lookup: Optional[Dict[str, float]] = None,
                 limit_to_se: bool = True) -> List[Dict]:
    """
    Build per-row score table for GSM8K.

    By default (limit_to_se=True), only keep questions for which a
    semantic-entropy value is also available, so all GSM8K scores in
    the cross-table are computed on the SAME set of questions. This
    makes the AUC of WEPR/EPR/SE directly comparable on GSM8K.

    Set limit_to_se=False to keep all 1 319 GSM8K rows (in which case
    the SE column will be sparse).
    """
    rows = []
    se_lookup = se_lookup or {}
    for it in items:
        if it.get("judge_correct") is None:
            continue
        td = it.get("token_data") or []
        if not td:
            continue
        qid = it.get("question_id")
        if limit_to_se and se_lookup and qid not in se_lookup:
            continue  # skip questions without SE -> aligned subset
        seq = seq_scores_from_token_data(td)
        row = {
            "dataset": "GSM8K",
            "correct": int(bool(it["judge_correct"])),
            **seq,
        }
        ep = epr_score(td)
        if ep is not None:
            row["epr"] = ep
        if wepr_model is not None:
            wp = wepr_with_model(td, wepr_model)
            if wp is not None:
                row["wepr"] = wp
        ans_td = it.get("answer_token_data") or []
        if ans_td:
            ans_seq = seq_scores_from_token_data(ans_td)
            for k, v in ans_seq.items():
                row[f"answer_{k}"] = v
        if qid in se_lookup:
            row["semantic_entropy"] = se_lookup[qid]
        rows.append(row)
    return rows


def _mmlu_letter(field) -> Optional[str]:
    """Robustly extract A/B/C/D letter from heterogeneous MMLU formats."""
    if field is None:
        return None
    if isinstance(field, str):
        s = field.strip().upper()
        return s if s in {"A", "B", "C", "D"} else None
    if isinstance(field, dict):
        # solution: {'answer_letter': 'C', ...}, model_answer: {'letter': 'C', ...}
        for k in ("answer_letter", "letter", "answer"):
            v = field.get(k)
            if isinstance(v, str):
                s = v.strip().upper()
                if s in {"A", "B", "C", "D"}:
                    return s
    return None


def scores_mmlu(items: List[Dict]) -> List[Dict]:
    rows = []
    for it in items:
        sol_letter = _mmlu_letter(it.get("solution"))
        ans_letter = _mmlu_letter(it.get("model_answer"))
        probs = it.get("probs_abcd") or {}
        if sol_letter is None or ans_letter is None or not probs:
            continue
        # Normalize probs: keys may be lower or upper case
        probs_norm = {str(k).strip().upper(): float(v)
                      for k, v in probs.items()
                      if str(k).strip().upper() in {"A", "B", "C", "D"}}
        if ans_letter not in probs_norm:
            continue
        correct = int(sol_letter == ans_letter)
        row = {
            "dataset": "MMLU",
            "correct": correct,
            "max_softmax": probs_norm[ans_letter],
        }
        # 4-way entropy as alt score
        p = np.array(list(probs_norm.values()), dtype=float)
        if p.sum() > 0:
            p = p / p.sum()
        ent = -float(sum(pi * math.log2(pi) for pi in p if pi > 0))
        row["mmlu_entropy"] = ent
        rows.append(row)
    return rows


# ──────────────────────── Score orientation ────────────────────────

# Some scores are "higher = more confident", others the opposite.
# For AUC we orient them so that high score => correct=1.
HIGHER_IS_BETTER = {
    # higher = more likely correct
    "confidence_verbalized": True,
    "judge_confidence":      True,
    "judge_accuracy_5":      True,
    "prob_joint_seq":        True,
    "prob_joint_all":        True,
    "prob_geo_mean_seq":     True,
    "prob_geo_mean_all":     True,
    "p_min_seq":             True,
    "p_min_all":             True,
    "wepr":                  True,
    "max_softmax":           True,
    # higher = LESS likely correct (distance / uncertainty)
    "perplexity_seq":        False,
    "perplexity_all":        False,
    "epr":                   False,
    "semantic_entropy":      False,
    "selfcheck":             False,
    "mmlu_entropy":          False,
    "logprob_mean_seq":      True,   # higher (less negative) = better
    # GSM8K answer-only mirrors of the seq variants
    "answer_prob_joint_seq":     True,
    "answer_prob_geo_mean_seq":  True,
    "answer_perplexity_seq":     False,
    "answer_p_min_seq":          True,
    "answer_logprob_mean_seq":   True,
}


PROBABILITY_LIKE = {
    # scores in [0,1] interpretable as P(correct) for ECE/Brier
    "confidence_verbalized", "judge_confidence",
    "prob_joint_seq", "prob_joint_all",
    "prob_geo_mean_seq", "prob_geo_mean_all",
    "p_min_seq", "p_min_all",
    "wepr", "max_softmax",
    "answer_prob_joint_seq", "answer_prob_geo_mean_seq",
    "answer_p_min_seq",
}


# ──────────────────────── Aggregation ────────────────────────

def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    """For each (dataset, score) compute n / AUC / ECE / Brier / Spearman."""
    if not rows:
        return []
    # Collect per-dataset score columns
    by_dataset: Dict[str, List[Dict]] = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    out = []
    for ds, items in by_dataset.items():
        # All columns appearing at least once
        cols = set()
        for r in items:
            cols.update(r.keys())
        cols.discard("dataset")
        cols.discard("correct")
        cols.discard("abs_error")
        cols.discard("n_tokens")
        for col in sorted(cols):
            ys, ss, errs = [], [], []
            for r in items:
                v = r.get(col)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                ys.append(r["correct"])
                ss.append(float(v))
                errs.append(r.get("abs_error"))
            if len(ys) < 5:
                continue
            y = np.array(ys, dtype=int)
            s = np.array(ss, dtype=float)

            higher_better = HIGHER_IS_BETTER.get(col, True)
            s_for_auc = s if higher_better else -s
            auc = safe_auc(y, s_for_auc)

            ece = brier = None
            if col in PROBABILITY_LIKE:
                p = s
                ece = expected_calibration_error(y, p, n_bins=10)
                brier = safe_brier(y, p)

            spearman_dist = None
            valid_errs = [e for e in errs if e is not None]
            if (col in HIGHER_IS_BETTER and ds == "TriviaQA-num"
                    and len(valid_errs) == len(s)):
                err_arr = np.array(valid_errs, dtype=float)
                # Confidence vs error: expect negative spearman if higher_better.
                signed = s if higher_better else -s
                spearman_dist = safe_spearman(signed, -err_arr)
                # Convention: positive = higher confidence ↔ closer to truth.

            out.append({
                "dataset": ds,
                "score": col,
                "n": int(len(s)),
                "auc": auc,
                "ece": ece,
                "brier": brier,
                "spearman_conf_vs_truth": spearman_dist,
            })
    return out


# ──────────────────────── Output writing ────────────────────────

def write_long_csv(path: Path, table: List[Dict]):
    if not table:
        path.write_text("")
        return
    cols = ["dataset", "score", "n", "auc", "ece", "brier",
            "spearman_conf_vs_truth"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in table:
            w.writerow({c: row.get(c) for c in cols})


def write_wide_auc(path: Path, table: List[Dict]):
    """Pivot AUC by score × dataset."""
    datasets = sorted({r["dataset"] for r in table})
    scores = sorted({r["score"] for r in table})
    matrix: Dict[Tuple[str, str], Optional[float]] = {}
    for r in table:
        matrix[(r["score"], r["dataset"])] = r["auc"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score", *datasets])
        for s in scores:
            row = [s]
            for d in datasets:
                v = matrix.get((s, d))
                row.append("" if v is None else f"{v:.4f}")
            w.writerow(row)


def write_latex(path: Path, table: List[Dict]):
    """Compact LaTeX longtable: AUC pivot + ECE pivot."""
    datasets = sorted({r["dataset"] for r in table})
    scores = sorted({r["score"] for r in table})

    def cell(r: Dict, k: str) -> str:
        v = r.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        return f"{v:.3f}"

    auc_map = {(r["score"], r["dataset"]): r for r in table}

    lines = []
    lines.append("% Auto-generated by cross_table.py — do not edit by hand.")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-dataset comparison of confidence scores"
                 r" — ROC-AUC of correct/incorrect discrimination.}")
    lines.append(r"\label{tab:cross_auc}")
    col_spec = "l" + "c" * len(datasets)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = " & ".join(["Score"] + [d.replace("_", r"\_")
                                     for d in datasets])
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for s in scores:
        row_cells = [s.replace("_", r"\_")]
        for d in datasets:
            r = auc_map.get((s, d))
            row_cells.append("--" if r is None else cell(r, "auc"))
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Second sub-table for ECE on probability-like scores
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-dataset comparison — Expected Calibration"
                 r" Error (10 bins). Only scores in $[0,1]$ are shown.}")
    lines.append(r"\label{tab:cross_ece}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    for s in scores:
        if s not in PROBABILITY_LIKE:
            continue
        row_cells = [s.replace("_", r"\_")]
        any_val = False
        for d in datasets:
            r = auc_map.get((s, d))
            if r is None or r.get("ece") is None:
                row_cells.append("--")
            else:
                row_cells.append(cell(r, "ece"))
                any_val = True
        if any_val:
            lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────── Main ────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triviaqa_num", type=str, default=str(TRIVIAQA_NUM))
    parser.add_argument("--triviaqa_wepr", type=str, default=str(TRIVIAQA_WEPR))
    parser.add_argument("--eli5_judged", type=str, default=str(ELI5_JUDGED))
    parser.add_argument("--eli5_se", type=str, default=str(ELI5_SE))
    parser.add_argument("--eli5_sc", type=str, default=str(ELI5_SC))
    parser.add_argument("--mmlu", type=str, default=str(MMLU))
    parser.add_argument("--gsm8k", type=str, default=str(GSM8K_JUDGED))
    parser.add_argument("--se_triviaqa_num", type=str,
                        default=str(SE_TRIVIAQA_NUM))
    parser.add_argument("--triviaqa_num_topk", type=str,
                        default=str(TRIVIAQA_NUM_TOPK))
    parser.add_argument("--eli5_topk", type=str, default=str(ELI5_TOPK))
    parser.add_argument("--se_gsm8k", type=str, default=str(SE_GSM8K))
    parser.add_argument("--wepr_model", type=str,
                        default=str(ROOT.parent / "WEPR" / "data"
                                    / "wepr_model.json"))
    parser.add_argument("--out_long", type=str, default=str(OUT_LONG))
    parser.add_argument("--out_wide", type=str, default=str(OUT_WIDE))
    parser.add_argument("--out_tex", type=str, default=str(OUT_TEX))
    parser.add_argument(
        "--gsm8k_full",
        action="store_true",
        help=(
            "Use ALL 1319 GSM8K rows (default keeps only those that also "
            "have a semantic-entropy value, so all GSM8K scores are "
            "computed on the same aligned subset)."
        ),
    )
    args = parser.parse_args()

    # Load WEPR model if available, used to score TriviaQA-WEPR and GSM8K
    wepr_model = None
    if Path(args.wepr_model).exists():
        with open(args.wepr_model, "r", encoding="utf-8") as f:
            wepr_model = json.load(f)
        print(f"Loaded WEPR model: {args.wepr_model}")

    # SE lookups
    se_tqa_lookup: Dict[str, float] = {}
    for it in load_jsonl(Path(args.se_triviaqa_num)):
        if "error" in it or "question" not in it:
            continue
        if "semantic_entropy" in it:
            se_tqa_lookup[it["question"]] = float(it["semantic_entropy"])
    print(f"SE TriviaQA lookup: {len(se_tqa_lookup)} entries")

    se_eli5_lookup: Dict[str, float] = {}
    for it in load_jsonl(Path(args.eli5_se)):
        if "semantic_entropy" in it and "question_id" in it:
            se_eli5_lookup[it["question_id"]] = float(it["semantic_entropy"])
    print(f"SE ELI5 lookup: {len(se_eli5_lookup)} entries")

    sc_eli5_lookup: Dict[str, float] = {}
    for it in load_jsonl(Path(args.eli5_sc)):
        if "selfcheck_score" in it and "question_id" in it:
            sc_eli5_lookup[it["question_id"]] = float(it["selfcheck_score"])
    print(f"SelfCheck ELI5 lookup: {len(sc_eli5_lookup)} entries")

    # New top-K lookups (used to compute EPR/WEPR on TriviaQA-num and ELI5)
    triviaqa_num_topk_lookup: Dict[str, Dict] = {}
    for it in load_jsonl(Path(args.triviaqa_num_topk)):
        if "error" in it:
            continue
        q = it.get("question")
        if q and (it.get("token_data") or []):
            triviaqa_num_topk_lookup[q] = it
    log.info(f"TriviaQA-num top-K lookup: "
             f"{len(triviaqa_num_topk_lookup)} entries")

    eli5_topk_lookup: Dict[str, Dict] = {}
    for it in load_jsonl(Path(args.eli5_topk)):
        if "error" in it:
            continue
        qid = it.get("question_id")
        if qid and (it.get("token_data") or []):
            eli5_topk_lookup[qid] = it
    log.info(f"ELI5 top-K lookup: {len(eli5_topk_lookup)} entries")

    # SE GSM8K lookup
    se_gsm8k_lookup: Dict[str, float] = {}
    for it in load_jsonl(Path(args.se_gsm8k)):
        if "error" in it:
            continue
        if "semantic_entropy" in it and "question_id" in it:
            se_gsm8k_lookup[it["question_id"]] = float(it["semantic_entropy"])
    log.info(f"SE GSM8K lookup: {len(se_gsm8k_lookup)} entries")

    # Per-dataset rows
    rows: List[Dict] = []
    log.info("Processing datasets...")

    log.info("Loading TriviaQA-num...")
    tqa_num = load_jsonl(Path(args.triviaqa_num))
    rows += scores_triviaqa_num(tqa_num, se_tqa_lookup,
                                 triviaqa_num_topk_lookup, wepr_model)
    log.info(f"  TriviaQA-num : {len(tqa_num)} loaded")

    log.info("Loading TriviaQA-WEPR (this can take a moment, ~74k rows)...")
    tqa_wepr = load_jsonl(Path(args.triviaqa_wepr))
    log.info(f"  TriviaQA-WEPR: {len(tqa_wepr)} loaded, computing scores...")
    rows += scores_triviaqa_wepr(tqa_wepr, wepr_model)

    log.info("Loading ELI5...")
    eli5 = load_jsonl(Path(args.eli5_judged))
    rows += scores_eli5(eli5, se_eli5_lookup, sc_eli5_lookup,
                         eli5_topk_lookup, wepr_model)
    log.info(f"  ELI5         : {len(eli5)} loaded")

    log.info("Loading GSM8K...")
    gsm8k = load_jsonl(Path(args.gsm8k))
    rows += scores_gsm8k(gsm8k, wepr_model, se_gsm8k_lookup,
                          limit_to_se=not args.gsm8k_full)
    if args.gsm8k_full:
        log.info(f"  GSM8K        : {len(gsm8k)} loaded "
                 "(full mode, sparse SE column)")
    else:
        log.info(f"  GSM8K        : {len(gsm8k)} loaded; "
                 f"keeping subset aligned with SE "
                 f"({len(se_gsm8k_lookup)} questions)")

    log.info("Loading MMLU...")
    mmlu = load_jsonl(Path(args.mmlu))
    rows += scores_mmlu(mmlu)
    log.info(f"  MMLU         : {len(mmlu)} loaded")

    log.info(f"Total rows assembled: {len(rows)} — aggregating metrics...")

    # Aggregate
    table = aggregate_rows(rows)

    # Write outputs
    out_long = Path(args.out_long)
    out_wide = Path(args.out_wide)
    out_tex = Path(args.out_tex)
    out_long.parent.mkdir(parents=True, exist_ok=True)

    write_long_csv(out_long, table)
    write_wide_auc(out_wide, table)
    write_latex(out_tex, table)

    print(f"\nWrote: {out_long}")
    print(f"Wrote: {out_wide}")
    print(f"Wrote: {out_tex}")

    # Pretty console summary
    print("\n=== AUC matrix preview ===")
    datasets = sorted({r["dataset"] for r in table})
    scores = sorted({r["score"] for r in table})
    print(f"{'score':35s} " + " ".join(f"{d:>16s}" for d in datasets))
    auc_map = {(r["score"], r["dataset"]): r for r in table}
    for s in scores:
        row = f"{s:35s} "
        for d in datasets:
            r = auc_map.get((s, d))
            row += f"{('--' if r is None or r.get('auc') is None else f'{r['auc']:.3f}'):>16s} "
        print(row)


if __name__ == "__main__":
    main()
