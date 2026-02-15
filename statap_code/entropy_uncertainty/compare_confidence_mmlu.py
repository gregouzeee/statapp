import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from statap_code.entropy_uncertainty.entropy_uncertainty import (
    confidence_from_entropy,
    normalize_probs,
    normalized_entropy,
)


CHOICES = ("A", "B", "C", "D")
DEFAULT_INPUTS = [
    "statap_code/Conformal_prediction/mmlu_500_confidence_probs.jsonl",
    "statap_code/Conformal_prediction/logit_mmlu_500_temp0_5.jsonl",
]
SCALAR_CONF_KEYS = {
    "declared_confidence",
    "self_confidence",
    "model_confidence",
    "confidence",
    "conf",
}
DIST_CONF_KEYS = {
    "confidence_abcd",
    "confidences_abcd",
    "confidence",
    "confidences",
    "confidence_probs",
    "declared_confidence_probs",
    "declared_confidence_abcd",
}


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_choice_dict(d: dict) -> Optional[np.ndarray]:
    if not isinstance(d, dict):
        return None
    if not all(k in d for k in CHOICES):
        return None
    vals = [_to_float(d[k]) for k in CHOICES]
    if any(v is None or math.isnan(v) for v in vals):
        return None
    return np.array(vals, dtype=float)


def _extract_choice_dict_from_raw_text(raw_text: str) -> Optional[np.ndarray]:
    if not isinstance(raw_text, str):
        return None
    txt = raw_text.strip()
    if not txt:
        return None

    # Try strict JSON parsing first
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            vec = _extract_choice_dict(obj)
            if vec is not None:
                return vec
    except Exception:
        pass

    # Fallback: locate an object-like substring and parse again
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                vec = _extract_choice_dict(obj)
                if vec is not None:
                    return vec
        except Exception:
            return None
    return None


def _extract_probs(rec: dict) -> Optional[np.ndarray]:
    for key in ("probs_abcd", "probs", "probabilities", "model_probs", "scores_abcd"):
        if key in rec:
            vec = _extract_choice_dict(rec[key]) if isinstance(rec[key], dict) else None
            if vec is not None:
                return normalize_probs(vec)

    # Nested fallback
    for key in ("model_answer", "solution", "metadata"):
        sub = rec.get(key)
        if isinstance(sub, dict):
            for sub_key in ("probs_abcd", "probs", "probabilities"):
                if sub_key in sub:
                    vec = _extract_choice_dict(sub[sub_key]) if isinstance(sub[sub_key], dict) else None
                    if vec is not None:
                        return normalize_probs(vec)
    return None


def _extract_true_letter(rec: dict) -> Optional[str]:
    if isinstance(rec.get("solution"), dict):
        sol = rec["solution"]
        ans = sol.get("answer_letter")
        if isinstance(ans, str) and ans.upper() in CHOICES:
            return ans.upper()
        idx = sol.get("answer_index")
        if isinstance(idx, int) and 0 <= idx < len(CHOICES):
            return CHOICES[idx]

    ans = rec.get("answer_letter")
    if isinstance(ans, str) and ans.upper() in CHOICES:
        return ans.upper()

    idx = rec.get("answer_index")
    if isinstance(idx, int) and 0 <= idx < len(CHOICES):
        return CHOICES[idx]
    return None


def _extract_pred_letter(rec: dict, probs: Optional[np.ndarray]) -> Optional[str]:
    if isinstance(rec.get("model_answer"), dict):
        ans = rec["model_answer"].get("letter")
        if isinstance(ans, str) and ans.upper() in CHOICES:
            return ans.upper()

    for key in ("pred", "prediction", "pred_letter", "predicted_letter"):
        val = rec.get(key)
        if isinstance(val, str) and val.upper() in CHOICES:
            return val.upper()

    if probs is not None:
        return CHOICES[int(np.argmax(probs))]
    return None


def _find_scalar_confidence(obj) -> Optional[float]:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in SCALAR_CONF_KEYS:
                    fv = _to_float(v)
                    if fv is not None:
                        return float(np.clip(fv, 0.0, 1.0 if fv <= 1.0 else 100.0))
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


def _find_distribution_confidence(rec: dict, pred: Optional[str]) -> Optional[float]:
    if pred is None:
        return None
    for key in DIST_CONF_KEYS:
        if key in rec and isinstance(rec[key], dict):
            vec = _extract_choice_dict(rec[key])
            if vec is not None:
                p = normalize_probs(vec)
                return float(p[CHOICES.index(pred)])
    return None


def _extract_declared_confidence(rec: dict, pred: Optional[str]) -> Optional[float]:
    raw = _find_scalar_confidence(rec)
    if raw is not None:
        # Interpret 0-100 as percentages if needed
        return float(raw / 100.0) if raw > 1.0 else float(raw)
    from_distribution = _find_distribution_confidence(rec, pred)
    if from_distribution is not None:
        return from_distribution

    # Common case in this project: model_answer.raw_text stores A/B/C/D confidences.
    if pred is not None and isinstance(rec.get("model_answer"), dict):
        raw_text = rec["model_answer"].get("raw_text")
        vec = _extract_choice_dict_from_raw_text(raw_text) if isinstance(raw_text, str) else None
        if vec is not None:
            p = normalize_probs(vec)
            return float(p[CHOICES.index(pred)])
    return None


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.zeros(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def metric_brier(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((s - y) ** 2))


def metric_ece(y: np.ndarray, s: np.ndarray, bins: int = 10) -> float:
    s = np.clip(s, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not np.any(mask):
            continue
        conf = np.mean(s[mask])
        acc = np.mean(y[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def metric_auroc(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average_ties(s)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def metric_auprc(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def accuracy_coverage_rows(y: np.ndarray, s: np.ndarray) -> List[Dict[str, float]]:
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    n = len(y_sorted)
    rows = []
    for cov in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        k = max(1, int(np.ceil(cov * n)))
        acc = float(np.mean(y_sorted[:k]))
        thr = float(s_sorted[k - 1])
        rows.append({"coverage": cov, "k": k, "accuracy": acc, "threshold": thr})
    return rows


def _nan_to_none(x: float):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def parse_jsonl(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            probs = _extract_probs(rec)
            true_letter = _extract_true_letter(rec)
            pred_letter = _extract_pred_letter(rec, probs)
            uid = rec.get("uid") if isinstance(rec.get("uid"), str) else f"{path.stem}:{line_no}"

            if probs is None or true_letter is None or pred_letter is None:
                continue

            y = 1 if pred_letter == true_letter else 0
            pmax = float(np.max(probs))
            p_pred = float(probs[CHOICES.index(pred_letter)])
            ent = float(normalized_entropy(probs))
            c_ent = float(confidence_from_entropy(probs))
            c_decl = _extract_declared_confidence(rec, pred_letter)
            full_logits_abcd = None
            if isinstance(rec.get("logits_abcd"), dict):
                full_logits_abcd = all(rec["logits_abcd"].get(k) is not None for k in CHOICES)

            out.append(
                {
                    "source": path.name,
                    "uid": uid,
                    "y": y,
                    "true": true_letter,
                    "pred": pred_letter,
                    "c_ent": c_ent,
                    "c_pmax": pmax,
                    "c_predprob": p_pred,
                    "c_decl": c_decl,
                    "entropy_norm": ent,
                    "full_logits_abcd": full_logits_abcd,
                }
            )
    return out


def evaluate(records: List[Dict[str, object]], ece_bins: int) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    y = np.array([int(r["y"]) for r in records], dtype=float)
    base_acc = float(np.mean(y))

    metrics: Dict[str, object] = {
        "n": int(len(records)),
        "accuracy_top1": base_acc,
        "signals": {},
    }
    coverage_rows: List[Dict[str, float]] = []

    for sig in ("c_decl", "c_ent", "c_pmax", "c_predprob"):
        s_full = np.array([np.nan if r[sig] is None else float(r[sig]) for r in records], dtype=float)
        mask = np.isfinite(s_full)
        if int(np.sum(mask)) < 5:
            continue
        y_sig = y[mask]
        s_sig = np.clip(s_full[mask], 0.0, 1.0)

        sig_metrics = {
            "n": int(len(y_sig)),
            "brier": _nan_to_none(metric_brier(y_sig, s_sig)),
            "ece": _nan_to_none(metric_ece(y_sig, s_sig, bins=ece_bins)),
            "auroc": _nan_to_none(metric_auroc(y_sig, s_sig)),
            "auprc": _nan_to_none(metric_auprc(y_sig, s_sig)),
        }
        metrics["signals"][sig] = sig_metrics

        for row in accuracy_coverage_rows(y_sig, s_sig):
            coverage_rows.append(
                {
                    "signal": sig,
                    "coverage": row["coverage"],
                    "k": row["k"],
                    "accuracy": row["accuracy"],
                    "threshold": row["threshold"],
                }
            )

    return metrics, coverage_rows


def write_per_item_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["source", "uid", "y", "true", "pred", "c_decl", "c_ent", "c_pmax", "c_predprob", "entropy_norm"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_coverage_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["source", "signal", "coverage", "k", "accuracy", "threshold"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _pick_source_name(names: List[str], contains: str) -> Optional[str]:
    lowered = contains.lower()
    for n in names:
        if lowered in n.lower():
            return n
    return None


def build_paired_rows(
    rows_logprob: List[Dict[str, object]], rows_decl: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    by_uid_logprob = {str(r["uid"]): r for r in rows_logprob}
    by_uid_decl = {str(r["uid"]): r for r in rows_decl}
    common_uids = sorted(set(by_uid_logprob.keys()) & set(by_uid_decl.keys()))

    paired: List[Dict[str, object]] = []
    for uid in common_uids:
        a = by_uid_logprob[uid]
        b = by_uid_decl[uid]
        paired.append(
            {
                "uid": uid,
                "true": a["true"],
                "pred_logprob": a["pred"],
                "pred_decl": b["pred"],
                "y_logprob": a["y"],
                "y_decl": b["y"],
                "c_ent_logprob": a["c_ent"],
                "c_ent_decl": b["c_ent"],
                "c_pmax_logprob": a["c_pmax"],
                "c_pmax_decl": b["c_pmax"],
                "c_decl": b["c_decl"],
                "full_logits_abcd_logprob": a.get("full_logits_abcd"),
            }
        )
    return paired


def write_paired_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "uid",
        "true",
        "pred_logprob",
        "pred_decl",
        "y_logprob",
        "y_decl",
        "c_ent_logprob",
        "c_ent_decl",
        "c_pmax_logprob",
        "c_pmax_decl",
        "c_decl",
        "full_logits_abcd_logprob",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_paired_coverage_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Coverage rows computed on paired data, using the matching target for each signal:
    - *_logprob signals -> y_logprob
    - *_decl signals    -> y_decl
    """
    signal_to_y = {
        "c_ent_logprob": "y_logprob",
        "c_pmax_logprob": "y_logprob",
        "c_ent_decl": "y_decl",
        "c_pmax_decl": "y_decl",
        "c_decl": "y_decl",
    }
    out: List[Dict[str, object]] = []
    for signal, y_col in signal_to_y.items():
        pairs: List[Tuple[float, int]] = []
        for r in rows:
            s = r.get(signal)
            y = r.get(y_col)
            if s is None or y is None:
                continue
            try:
                sf = float(s)
                yf = int(y)
            except Exception:
                continue
            if not np.isfinite(sf):
                continue
            sf = float(np.clip(sf, 0.0, 1.0))
            pairs.append((sf, yf))

        if len(pairs) == 0:
            continue

        pairs.sort(key=lambda x: x[0], reverse=True)
        scores = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)
        n = len(pairs)
        for cov in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k = max(1, int(np.ceil(cov * n)))
            acc = float(np.mean(ys[:k]))
            thr = float(scores[k - 1])
            out.append(
                {
                    "signal": signal,
                    "target": y_col,
                    "coverage": cov,
                    "k": k,
                    "accuracy": acc,
                    "threshold": thr,
                    "n_total": n,
                }
            )
    return out


def write_paired_coverage_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["signal", "target", "coverage", "k", "accuracy", "threshold", "n_total"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description="Compare confidence signals on MMLU JSONL files: c_decl vs c_ent vs c_pmax."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help=(
            "One or more JSONL files (e.g. confidence file and logprobs file). "
            "If omitted, uses built-in defaults for MMLU 500."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="statap_code/entropy_uncertainty/results",
        help="Directory for paired CSV output.",
    )
    parser.add_argument("--prefix", default="mmlu_conf_compare", help="Prefix for output files.")
    parser.add_argument(
        "--include-partial-logits",
        action="store_true",
        help=(
            "Keep rows even if some A/B/C/D logits are missing in the logprob source. "
            "By default, only complete logits are kept."
        ),
    )
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = args.inputs if args.inputs else DEFAULT_INPUTS

    rows_by_source: Dict[str, List[Dict[str, object]]] = {}
    report: Dict[str, object] = {"inputs": [str(Path(p)) for p in inputs], "sources": {}}

    for input_path in inputs:
        p = (PROJECT_ROOT / input_path).resolve() if not Path(input_path).is_absolute() else Path(input_path)
        if not p.exists():
            report["sources"][input_path] = {"error": "file_not_found"}
            continue

        rows = parse_jsonl(p)
        if not rows:
            report["sources"][p.name] = {"error": "no_valid_rows"}
            continue

        rows_by_source[p.name] = rows
        report["sources"][p.name] = {"n": int(len(rows))}

    paired_path = out_dir / f"{args.prefix}_paired.csv"
    coverage_path = out_dir / f"{args.prefix}_paired_coverage.csv"

    paired_rows: List[Dict[str, object]] = []
    source_names = list(rows_by_source.keys())
    logprob_source = _pick_source_name(source_names, "logit")
    decl_source = _pick_source_name(source_names, "confidence")
    if logprob_source is not None and decl_source is not None:
        paired_rows = build_paired_rows(rows_by_source[logprob_source], rows_by_source[decl_source])
    elif len(source_names) >= 2:
        # Fallback: pair first two sources in input order.
        paired_rows = build_paired_rows(rows_by_source[source_names[0]], rows_by_source[source_names[1]])

    n_before_filter = len(paired_rows)
    if not args.include_partial_logits:
        paired_rows = [r for r in paired_rows if r.get("full_logits_abcd_logprob") is True]
    n_after_filter = len(paired_rows)

    if not paired_rows:
        print("[error] unable to pair sources (need two compatible JSONL files).")
        for source_name, source_report in report["sources"].items():
            print(f"- {source_name}: {source_report}")
        return

    write_paired_csv(paired_path, paired_rows)
    coverage_rows = build_paired_coverage_rows(paired_rows)
    write_paired_coverage_csv(coverage_path, coverage_rows)

    if paired_rows:
        print(f"[ok] paired:   {paired_path}")
        print(f"[ok] coverage: {coverage_path}")
        if not args.include_partial_logits:
            print(f"- filter_full_logits: kept {n_after_filter}/{n_before_filter}")
    for source_name, source_report in report["sources"].items():
        if "error" in source_report:
            print(f"- {source_name}: error={source_report['error']}")
            continue
        print(f"- {source_name}: n={source_report['n']}")


if __name__ == "__main__":
    main()
