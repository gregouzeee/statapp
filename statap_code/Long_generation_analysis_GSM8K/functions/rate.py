import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = "statap_code/gsm8k/cleaned_answers_logprob.jsonl"
OUTPUT_FEATURES_CSV = "statap_code/gsm8k/logprob_features_interpretable.csv"


# =========================================================
# 1. Accuracy helpers
# =========================================================

def extract_last_number(text):
    if text is None:
        return None
    text = str(text).strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def normalize_answer(ans):
    if ans is None:
        return None
    num = extract_last_number(ans)
    if num is not None:
        return num
    return str(ans).strip().lower()


def is_correct(model_ans, gold_ans):
    m = normalize_answer(model_ans)
    g = normalize_answer(gold_ans)

    if m is None or g is None:
        return False

    if isinstance(m, float) and isinstance(g, float):
        return abs(m - g) < 1e-9

    return m == g


# =========================================================
# 2. Small helpers
# =========================================================

def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else np.nan


def safe_std(x):
    return float(np.std(x)) if len(x) > 0 else np.nan


def safe_min(x):
    return float(np.min(x)) if len(x) > 0 else np.nan


def safe_max(x):
    return float(np.max(x)) if len(x) > 0 else np.nan


def safe_median(x):
    return float(np.median(x)) if len(x) > 0 else np.nan


def safe_slope(x):
    if len(x) < 2:
        return np.nan
    y = np.asarray(x, dtype=float)
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t, y, 1)[0])


def longest_run(mask):
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def segment_mean(arr):
    arr = np.asarray(arr, dtype=float)
    return safe_mean(arr)


# =========================================================
# 3. Token-level quantities
# =========================================================

def compute_step_entropy(step):
    if not step:
        return np.nan

    lps = np.array([c["logprob"] for c in step], dtype=float)
    probs = np.exp(lps - np.max(lps))
    s = probs.sum()
    if s <= 0:
        return np.nan
    probs = probs / s
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def extract_features_from_logprobs(logprobs):
    if not logprobs:
        return None

    top1 = []
    entropies = []
    margins = []
    candidate_counts = []

    for step in logprobs:
        if not step:
            continue

        candidate_counts.append(len(step))

        lps = np.array([c["logprob"] for c in step], dtype=float)

        top1_lp = float(step[0]["logprob"])
        top1.append(top1_lp)

        if len(step) >= 2:
            top2_lp = float(step[1]["logprob"])
            margins.append(top1_lp - top2_lp)

        probs = np.exp(lps - np.max(lps))
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs = probs / probs_sum

        entropies.append(compute_step_entropy(step))

    if len(top1) == 0:
        return None

    top1 = np.asarray(top1, dtype=float)
    entropies = np.asarray(entropies, dtype=float)
    margins = np.asarray(margins, dtype=float) if len(margins) > 0 else np.array([], dtype=float)
    candidate_counts = np.asarray(candidate_counts, dtype=float)

    n = len(top1)
    third = max(1, n // 3)

    top1_first = top1[:third]
    top1_last = top1[2 * third:] if n >= 3 else top1[third:]

    entropy_first = entropies[:third]
    entropy_last = entropies[2 * third:] if n >= 3 else entropies[third:]

    low_lp_mask = top1 < -2.0
    very_low_lp_mask = top1 < -3.0

    feats = {
        # longueur
        "length": n,

        # confiance moyenne
        "top1_mean": safe_mean(top1),
        "top1_std": safe_std(top1),
        "top1_min": safe_min(top1),

        # stabilité / fluctuations
        "top1_slope": safe_slope(top1),

        # tokens très incertains
        "ratio_lp_below_-2": float(np.mean(low_lp_mask)),
        "ratio_lp_below_-3": float(np.mean(very_low_lp_mask)),
        "longest_run_lp_below_-2": longest_run(low_lp_mask),

        # dispersion de la distribution
        "entropy_mean": safe_mean(entropies),
        "entropy_std": safe_std(entropies),
        "entropy_max": safe_max(entropies),

        # séparation top1 / top2
        "margin_mean": safe_mean(margins),
        "margin_std": safe_std(margins),
        "margin_min": safe_min(margins),

        # structure du top-k disponible
        "candidate_count_mean": safe_mean(candidate_counts),

        # dynamique début / fin
        "top1_first_third_mean": segment_mean(top1_first),
        "top1_last_third_mean": segment_mean(top1_last),
        "top1_last_minus_first_third_mean": (
            segment_mean(top1_last) - segment_mean(top1_first)
            if len(top1_first) > 0 and len(top1_last) > 0 else np.nan
        ),

        "entropy_first_third_mean": segment_mean(entropy_first),
        "entropy_last_third_mean": segment_mean(entropy_last),
        "entropy_last_minus_first_third_mean": (
            segment_mean(entropy_last) - segment_mean(entropy_first)
            if len(entropy_first) > 0 and len(entropy_last) > 0 else np.nan
        ),
    }

    return feats


def prefix_features(feats, prefix):
    return {f"{prefix}_{k}": v for k, v in feats.items()}


def get_logprobs_by_mode(obj, mode):
    reasoning = obj.get("reasoning_logprobs", []) or []
    final = obj.get("final_answer_logprobs", []) or []

    if mode == "reasoning":
        return reasoning
    if mode == "final":
        return final
    if mode == "all":
        return reasoning + final

    raise ValueError(f"Mode inconnu: {mode}")


# =========================================================
# 4. Build dataframe
# =========================================================

def load_feature_dataframe(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            correct = is_correct(obj.get("model_answer"), obj.get("gold_answer"))

            row = {
                "row_index": i,
                "id": obj.get("id", i),
                "model_answer": obj.get("model_answer"),
                "gold_answer": obj.get("gold_answer"),
                "correct": int(correct),
                "group": "correct" if correct else "wrong",
            }

            has_any_feature = False

            for mode in ["reasoning", "final", "all"]:
                lp = get_logprobs_by_mode(obj, mode)
                feats = extract_features_from_logprobs(lp)

                if feats is not None:
                    row.update(prefix_features(feats, mode))
                    has_any_feature = True
                else:
                    row[f"{mode}_length"] = 0

            if has_any_feature:
                rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# 5. Summary by group
# =========================================================

def summarize_groups(df):
    numeric_cols = [
        c for c in df.columns
        if c not in {"row_index", "id", "model_answer", "gold_answer", "group"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print("\n==============================")
    print("GLOBAL COUNTS")
    print("==============================")
    print(df["group"].value_counts(dropna=False))
    print()
    print(f"Accuracy: {df['correct'].mean() * 100:.2f}%")

    means = df.groupby("group")[numeric_cols].mean().T
    stds = df.groupby("group")[numeric_cols].std().T
    medians = df.groupby("group")[numeric_cols].median().T

    print("\n==============================")
    print("GROUP MEANS")
    print("==============================")
    print(means)

    print("\n==============================")
    print("GROUP STDS")
    print("==============================")
    print(stds)

    print("\n==============================")
    print("GROUP MEDIANS")
    print("==============================")
    print(medians)

    if "wrong" in means.columns and "correct" in means.columns:
        print("\n==============================")
        print("MEAN DIFFERENCE (wrong - correct)")
        print("==============================")
        diff = (means["wrong"] - means["correct"]).sort_values(ascending=False)
        print(diff)

    if "wrong" in means.columns and "correct" in means.columns:
        print("\n==============================")
        print("ABSOLUTE MEAN DIFFERENCE")
        print("==============================")
        absdiff = (means["wrong"] - means["correct"]).abs().sort_values(ascending=False)
        print(absdiff)


# =========================================================
# 6. Main
# =========================================================
def save_summary_html(summary_df, path="statap_code/gsm8k/logprob_features_summary.html"):
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Feature Summary</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1 {{
                margin-bottom: 16px;
            }}
            .table-container {{
                width: 100%;
                max-width: 100%;
                overflow-x: auto;
                overflow-y: auto;
                max-height: 80vh;
                border: 1px solid #ccc;
            }}
            table {{
                border-collapse: collapse;
                min-width: 1200px;
                width: max-content;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 10px;
                text-align: right;
                white-space: nowrap;
            }}
            th {{
                background-color: #f4f4f4;
                position: sticky;
                top: 0;
                z-index: 2;
            }}
            td:first-child, th:first-child {{
                text-align: left;
                position: sticky;
                left: 0;
                background-color: #fafafa;
                z-index: 3;
            }}
            tr:nth-child(even) {{
                background-color: #fcfcfc;
            }}
        </style>
    </head>
    <body>
        <h1>Logprob Feature Summary</h1>
        <div class="table-container">
            {summary_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
        </div>
    </body>
    </html>
    """

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML summary saved to: {path}")

def main():
    if not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_PATH}")

    df = load_feature_dataframe(INPUT_PATH)

    if len(df) == 0:
        print("Aucune ligne exploitable.")
        return

    df.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8")
    print(f"Features saved to: {OUTPUT_FEATURES_CSV}")

    summarize_groups(df)

    print("\n==============================")
    print("TOP FEATURES TO INSPECT FIRST")
    print("==============================")
    suggested = [
        "reasoning_top1_mean",
        "reasoning_top1_std",
        "reasoning_top1_min",
        "reasoning_worst_pos_norm",
        "reasoning_diff_std",
        "reasoning_absdiff_mean",
        "reasoning_ratio_lp_below_-2",
        "reasoning_ratio_lp_below_-3",
        "reasoning_longest_run_lp_below_-2",
        "reasoning_entropy_mean",
        "reasoning_entropy_std",
        "reasoning_entropy_max",
        "reasoning_margin_mean",
        "reasoning_margin_min",
        "reasoning_maxprob_topk_mean",
        "reasoning_gini_topk_mean",
        "reasoning_top1_last_minus_first_third_mean",
        "reasoning_entropy_last_minus_first_third_mean",
        "reasoning_length",
        "final_top1_mean",
        "final_top1_min",
        "final_entropy_mean",
        "final_margin_mean",
        "final_length",
        "all_top1_mean",
        "all_top1_std",
        "all_entropy_mean",
        "all_margin_mean",
        "all_length",
    ]

    rows = []
    for feat in suggested:
        if feat not in df.columns:
            continue

        correct_vals = df.loc[df["group"] == "correct", feat].dropna()
        wrong_vals = df.loc[df["group"] == "wrong", feat].dropna()

        rows.append({
            "feature": feat,
            "correct_mean": correct_vals.mean() if len(correct_vals) else np.nan,
            "correct_std": correct_vals.std() if len(correct_vals) else np.nan,
            "wrong_mean": wrong_vals.mean() if len(wrong_vals) else np.nan,
            "wrong_std": wrong_vals.std() if len(wrong_vals) else np.nan,
            "diff_wrong_minus_correct": (
                wrong_vals.mean() - correct_vals.mean()
                if len(correct_vals) and len(wrong_vals) else np.nan
            ),
        })

    summary_df = pd.DataFrame(rows)
    summary_df["abs_diff"] = summary_df["diff_wrong_minus_correct"].abs()
    summary_df = summary_df.sort_values("abs_diff", ascending=False)

    print(summary_df.head(12).round(4))

    summary_df.to_csv("statap_code/gsm8k/logprob_features_summary.csv", index=False, encoding="utf-8")
    print("Compact summary saved to: statap_code/gsm8k/logprob_features_summary.csv")

    save_summary_html(summary_df, "statap_code/gsm8k/logprob_features_summary.html")


if __name__ == "__main__":
    main()