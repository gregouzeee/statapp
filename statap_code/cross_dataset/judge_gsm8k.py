"""
GSM8K — Annotate generated answers as correct / incorrect
==========================================================

Input:  data/gsm8k_topk.jsonl  (output of generate_gsm8k_topk.py)
Output: data/gsm8k_judged.jsonl  (same structure + 'judge_correct' bool)

Annotation is purely numeric: we compare the parsed `answer_value`
against the gold `ground_truth` from the GSM8K dataset, with a relative
tolerance to absorb rounding artefacts. No LLM judge is used (gold is
unambiguous on GSM8K).

The output file uses the same field name `judge_correct` as the WEPR
training script expects, so train_wepr.py works without modification.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("judge_gsm8k")


ROOT = Path(__file__).resolve().parent
IN_JSONL = ROOT / "data" / "gsm8k_topk.jsonl"
OUT_JSONL = ROOT / "data" / "gsm8k_judged.jsonl"


def is_correct(pred: Optional[float], gt: Optional[float],
               abs_tol: float = 1e-3, rel_tol: float = 1e-3) -> bool:
    if pred is None or gt is None:
        return False
    if math.isnan(pred) or math.isnan(gt):
        return False
    return math.isclose(pred, gt, abs_tol=abs_tol, rel_tol=rel_tol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(IN_JSONL))
    parser.add_argument("--output", type=str, default=str(OUT_JSONL))
    parser.add_argument("--abs_tol", type=float, default=1e-3)
    parser.add_argument("--rel_tol", type=float, default=1e-3)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading {in_path}")
    n_total = n_err = n_correct = n_parsed = 0
    with open(in_path, "r", encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as g:
        for line in f:
            if n_total and n_total % 500 == 0:
                log.info(f"  parsed {n_total} rows... "
                         f"running acc={n_correct / max(1, n_total - n_err):.3f}")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if "error" in obj:
                obj["judge_correct"] = None
                g.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_err += 1
                continue

            pred = obj.get("answer_value")
            gt = obj.get("ground_truth")
            if pred is not None:
                n_parsed += 1
            ok = is_correct(pred, gt,
                            abs_tol=args.abs_tol, rel_tol=args.rel_tol)
            obj["judge_correct"] = bool(ok)
            if ok:
                n_correct += 1
            g.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    print(f"Entries     : {n_total}")
    print(f"Errors      : {n_err}")
    print(f"Numeric ok  : {n_parsed}")
    print(f"Correct     : {n_correct}")
    valid = n_total - n_err
    if valid:
        print(f"Accuracy    : {n_correct / valid:.3%} (over non-error rows)")


if __name__ == "__main__":
    main()
