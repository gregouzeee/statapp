import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from statap_code.Conformal_prediction.Previous_version.conformal_prediction_tools import ConformalPredictor, AlphaTuner

LABELS_STR = ["A", "B", "C", "D"]


# ---------------------- CONFIG ----------------------
IN_JSONL = Path("statap_code/Conformal_prediction/logit_mmlu_500.jsonl")         
OUT_JSONL = Path("statap_code/Conformal_prediction/mmlu_500_conformal_output.jsonl")


# Split + tuning
random_seed = 42
split_cal_ratio = 0.5
val_ratio_of_remaining = 0.5

alphas = np.linspace(0.01, 0.5, 50)
coverage_target = 0.95
loss_lambdas = [1.0, 5.0, 20.0]
# ---------------------------------------------------


def split_three_ways(n_total: int, cal_ratio: float, val_ratio_of_remaining: float, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    n_cal = int(np.round(n_total * cal_ratio))
    remaining = n_total - n_cal
    n_val = int(np.round(remaining * val_ratio_of_remaining))
    n_test = remaining - n_val

    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal:n_cal + n_val]
    test_idx = perm[n_cal + n_val:]
    return cal_idx, val_idx, test_idx


def read_jsonl_examples(path: Path) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    examples_by_idx: Dict[int, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {
        "type": "meta",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": str(path),
        "label_meaning": "answer_index is 0=A,1=B,2=C,3=D.",
    }

    with path.open("r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples_by_idx[idx] = obj
            idx += 1

    meta["n_samples"] = int(len(examples_by_idx))
    return meta, examples_by_idx


def write_output_jsonl(path: Path, meta: Dict[str, Any], examples_by_idx: Dict[int, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for idx in sorted(examples_by_idx.keys()):
            f.write(json.dumps(examples_by_idx[idx], ensure_ascii=False) + "\n")


def parse_label(ex: Dict[str, Any]) -> Optional[int]:
    sol = ex.get("solution")
    if isinstance(sol, dict) and "answer_index" in sol:
        try:
            return int(sol["answer_index"])
        except Exception:
            return None
    if "answer_index" in ex:
        try:
            return int(ex["answer_index"])
        except Exception:
            return None
    return None


def probs_from_example(ex: Dict[str, Any]) -> np.ndarray:
    """
    Prefer probs_abcd.
    Else if logits_abcd exist (log-sum-exp masses), exp+normalize over A-D.
    Else uniform.
    """
    p = ex.get("probs_abcd")
    if isinstance(p, dict) and all(k in p for k in ("A", "B", "C", "D")):
        arr = np.array([float(p["A"]), float(p["B"]), float(p["C"]), float(p["D"])], dtype=float)
        s = arr.sum()
        return arr / s if s > 0 else np.ones(4) / 4

    lg = ex.get("logits_abcd")
    if isinstance(lg, dict):
        masses = []
        for k in ("A", "B", "C", "D"):
            v = lg.get(k)
            if v is None:
                masses.append(0.0)
            else:
                masses.append(float(np.exp(float(v))))
        arr = np.array(masses, dtype=float)
        s = arr.sum()
        return arr / s if s > 0 else np.ones(4) / 4

    return np.ones(4) / 4


def main():
    meta, examples_by_idx = read_jsonl_examples(IN_JSONL)

    # Build probs_mat and labels
    n_total = len(examples_by_idx)
    probs_mat = np.zeros((n_total, 4), dtype=float)
    labels = np.full((n_total,), -1, dtype=int)

    for i in range(n_total):
        ex = examples_by_idx[i]
        probs_mat[i] = probs_from_example(ex)
        y = parse_label(ex)
        labels[i] = y if y is not None else -1

    # drop unlabeled (should be none for MMLU)
    keep = np.where(labels >= 0)[0]
    if len(keep) != n_total:
        probs_mat = probs_mat[keep]
        labels = labels[keep]
        new_examples = {}
        for j, old_i in enumerate(keep.tolist()):
            new_examples[j] = examples_by_idx[int(old_i)]
        examples_by_idx = new_examples
        n_total = len(examples_by_idx)

    # Split cal/val/test
    cal_idx, val_idx, test_idx = split_three_ways(n_total, split_cal_ratio, val_ratio_of_remaining, random_seed)
    cal_set = set(map(int, cal_idx))
    val_set = set(map(int, val_idx))

    probs_cal, y_cal = probs_mat[cal_idx], labels[cal_idx]
    probs_val, y_val = probs_mat[val_idx], labels[val_idx]

    # ---------- Alpha tuning (APS + LAC) ----------
    tuner_aps = AlphaTuner("aps")
    grid_aps = tuner_aps.run_grid(probs_cal, y_cal, probs_val, y_val, alphas, loss_lambdas=loss_lambdas)
    sel_aps = tuner_aps.select_alphas(grid_aps, coverage_target=coverage_target)

    tuner_lac = AlphaTuner("lac")
    grid_lac = tuner_lac.run_grid(probs_cal, y_cal, probs_val, y_val, alphas, loss_lambdas=loss_lambdas)
    sel_lac = tuner_lac.select_alphas(grid_lac, coverage_target=coverage_target)

    print("\nAPS optimal alphas:")
    for k, v in tuner_aps.summarize_alphas(sel_aps).items():
        print(f"  {k}: {v}")

    print("\nLAC optimal alphas:")
    for k, v in tuner_lac.summarize_alphas(sel_lac).items():
        print(f"  {k}: {v}")

    # choose final alpha rule (same as before)
    alpha_aps = sel_aps.alpha_cov_max 
    alpha_lac = sel_lac.alpha_cov_max 

    # recalibrate on cal+val
    combined_idx = np.concatenate([cal_idx, val_idx]) if len(val_idx) else cal_idx
    probs_comb, y_comb = probs_mat[combined_idx], labels[combined_idx]

    cp_aps = ConformalPredictor("aps", alpha_aps).calibrate(probs_comb, y_comb)
    cp_lac = ConformalPredictor("lac", alpha_lac).calibrate(probs_comb, y_comb)

    # Evaluate on test
    cov_aps, size_aps = cp_aps.evaluate_sets(probs_mat[test_idx], labels[test_idx])
    cov_lac, size_lac = cp_lac.evaluate_sets(probs_mat[test_idx], labels[test_idx])

    # ---------- Meta enrichment ----------
    meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["split"] = {
        "split_cal_ratio": float(split_cal_ratio),
        "val_ratio_of_remaining": float(val_ratio_of_remaining),
        "n_total": int(n_total),
        "n_cal": int(len(cal_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(random_seed),
    }
    meta["alpha_tuning"] = {
        "coverage_target": float(coverage_target),
        "loss_lambdas": [float(x) for x in loss_lambdas],
        "aps": {"grid": grid_aps.to_dict(), "selection": sel_aps.to_dict()},
        "lac": {"grid": grid_lac.to_dict(), "selection": sel_lac.to_dict()},
    }
    meta["final"] = {
        "alpha_used": {"aps": float(alpha_aps), "lac": float(alpha_lac)},
        "q_hat": {"aps": float(cp_aps.q_hat), "lac": float(cp_lac.q_hat)},
        "test_eval": {
            "aps": {"coverage": float(cov_aps), "avg_size": float(size_aps)},
            "lac": {"coverage": float(cov_lac), "avg_size": float(size_lac)},
        },
    }

    # ---------- Enrich examples ----------
    for i in range(n_total):
        ex = examples_by_idx[i]

        split = "cal" if i in cal_set else ("val" if i in val_set else "test")
        ex["split"] = split

        p = probs_mat[i]
        y = int(labels[i])
        pred_idx = int(np.argmax(p))

        C_lac = cp_lac.predict(p)
        C_aps = cp_aps.predict(p)

        ex["true"] = LABELS_STR[y]
        ex["pred"] = LABELS_STR[pred_idx]

        ex["lac"] = {
            "set": [LABELS_STR[int(k)] for k in C_lac],
            "covered": bool(y in C_lac),
            "size": int(len(C_lac)),
            "p_true": float(p[y]),
        }
        ex["aps"] = {
            "set": [LABELS_STR[int(k)] for k in C_aps],
            "covered": bool(y in C_aps),
            "size": int(len(C_aps)),
            "p_true": float(p[y]),
        }

    write_output_jsonl(OUT_JSONL, meta=meta, examples_by_idx=examples_by_idx)
    print("DONE ->", OUT_JSONL)


if __name__ == "__main__":
    main()