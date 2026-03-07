"""
LLM-as-judge evaluation + comparison of importance methods
==========================================================
1.  Load existing results from triviaqa_results.jsonl
2.  Use Gemini (Vertex AI) as a judge to evaluate answer correctness
    (batched calls to minimize API usage)
3.  Compare token-level and word-level importance metrics
    between correct and incorrect answers
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ──────────────────────────── Config ────────────────────────────

JUDGE_MODEL = "gemini-2.0-flash"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256
JUDGE_BATCH_SIZE = 10     # questions per API call
MAX_RETRIES = 3

RESULTS_JSONL = Path(__file__).resolve().parent / "triviaqa_results.jsonl"
JUDGED_JSONL  = Path(__file__).resolve().parent / "triviaqa_judged.jsonl"
FIGS_DIR      = Path(__file__).resolve().parent / "figures"


# ──────────────────── Load existing results ─────────────────────

def load_results(path: Path) -> List[Dict]:
    """Load valid (non-error) results from the JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                continue
            results.append(obj)
    return results


# ──────────────────── LLM-as-judge (batched) ────────────────────

def _build_judge_prompt(batch: List[Dict]) -> str:
    """
    Build a batched prompt for the judge.
    The judge sees each question + ground truths + LLM answer
    and returns a JSON array of true/false.
    """
    lines = [
        "You are a strict factuality judge for trivia questions.",
        "",
        "RULES:",
        "- An answer is CORRECT (true) ONLY if it explicitly contains or directly",
        "  matches one of the ACCEPTED ANSWERS listed below.",
        "- Do NOT use your own knowledge. If the LLM answer gives a name, date,",
        "  or fact that is NOT in the accepted answers list, it is INCORRECT,",
        "  even if you know it to be factually related or equivalent.",
        "- Minor spelling variations (e.g. capitalization, punctuation) are OK.",
        "- Synonyms, aliases, or related facts NOT in the accepted list => false.",
        "",
        "Return ONLY a JSON object in this exact format:",
        '{"judgments": [true, false, true, ...]}',
        f"The array MUST have exactly {len(batch)} elements, in the same order.",
        "No explanations, no markdown fences, just the JSON.",
        "",
    ]
    for i, item in enumerate(batch):
        gt_str = " | ".join(item["ground_truths"][:8])
        lines.append(f"--- Item {i+1} ---")
        lines.append(f"Question: {item['question']}")
        lines.append(f"Accepted answers: {gt_str}")
        lines.append(f"LLM answer: {item['full_answer']}")
        lines.append("")

    return "\n".join(lines)


def _parse_judgments(raw_text: str, expected_len: int) -> Optional[List[Optional[bool]]]:
    """Parse the judge's JSON response into a list of booleans."""
    if not raw_text:
        return None

    # Strip markdown fences
    cleaned = re.sub(r"^(```json|```|''')\s*", "", raw_text.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        print(f"    [judge] WARNING: No JSON found in response: {raw_text[:200]}")
        return None

    try:
        data = json.loads(m.group(0))
        arr = data.get("judgments", None)
        if not isinstance(arr, list):
            print(f"    [judge] WARNING: No 'judgments' list in JSON: {data}")
            return None

        out: List[Optional[bool]] = []
        for x in arr[:expected_len]:
            if isinstance(x, bool):
                out.append(x)
            elif isinstance(x, str):
                xl = x.strip().lower()
                if xl in ("true", "yes"):
                    out.append(True)
                elif xl in ("false", "no"):
                    out.append(False)
                else:
                    out.append(None)
            else:
                out.append(None)

        # Pad if too short
        while len(out) < expected_len:
            out.append(None)
        return out

    except Exception as e:
        print(f"    [judge] WARNING: JSON parse error: {e}")
        return None


def judge_answers_batch(
    client,
    results: List[Dict],
    batch_size: int = JUDGE_BATCH_SIZE,
) -> List[Optional[bool]]:
    """
    Call the LLM judge on all results in batches.
    Returns a list of bool (True=correct, False=incorrect, None=parse error).
    """
    judgments: List[Optional[bool]] = []
    total = len(results)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = results[start:end]
        batch_num = start // batch_size + 1
        total_batches = math.ceil(total / batch_size)

        print(f"\n  [judge] Batch {batch_num}/{total_batches} (items {start+1}-{end}/{total})")

        prompt = _build_judge_prompt(batch)

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=JUDGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=JUDGE_TEMPERATURE,
                        max_output_tokens=JUDGE_MAX_TOKENS,
                        candidate_count=1,
                    ),
                )
                raw = (resp.text or "").strip()
                parsed = _parse_judgments(raw, expected_len=len(batch))
                if parsed is not None:
                    break
                print(f"    [judge] Parse failed on attempt {attempt}, retrying...")
            except Exception as e:
                print(f"    [judge] API error on attempt {attempt}: {e}")
                time.sleep(min(2.0 * attempt, 10.0))

        if parsed is None:
            print(f"    [judge] All retries failed — defaulting to None for this batch")
            parsed = [None] * len(batch)

        # Print batch results
        for i, (item, judgment) in enumerate(zip(batch, parsed)):
            status = "CORRECT" if judgment is True else "INCORRECT" if judgment is False else "UNKNOWN"
            print(f"    [{start+i+1}] {status:10s}  Q: {item['question'][:60]}...")
            print(f"                     A: {item['full_answer'][:60]}...")

        judgments.extend(parsed)

    return judgments


# ──────────── LLM-as-judge for token matching ─────────────────────

def _build_token_match_prompt(batch: List[Dict]) -> str:
    """
    Build a batched prompt for binary token matching: strict / none.
    """
    lines = [
        "You are a spelling/orthography comparison tool. You do NOT answer trivia.",
        "You do NOT use world knowledge. You ONLY compare character strings.",
        "",
        "TASK: For each item below you receive ACCEPTED ANSWERS and TOP TOKENS.",
        "Classify whether any top token orthographically matches an accepted answer.",
        "",
        "TWO LEVELS:",
        "",
        '"strict" — At least one top token is orthographically part of an',
        "   accepted answer. This means the token (ignoring case, accents,",
        "   hyphens, spaces) is a substring of an accepted answer OR the",
        "   accepted answer is a substring of the token, AND the token is a",
        "   meaningful word (not a stop word like 'in', 'the', 'of', 'a',",
        "   'and', 'is', 'was', 'it', 'to', 'for', 'on', 'at', 'by').",
        "   Examples:",
        "     - 'Einstein' vs 'Albert Einstein' => strict (substring)",
        "     - 'beekeeper' vs 'bee keeper' => strict (same word, spacing differs)",
        "     - '1889' vs '1889' => strict (identical)",
        "     - 'Stere' vs 'Stereo records' => strict (substring of answer)",
        "     - 'in' vs 'Nineteen-thirties' => none (stop word)",
        "     - 'first' vs '1930s' => none (no spelling overlap)",
        "",
        '"none" — No orthographic match at all.',
        "",
        "Return ONLY a JSON object:",
        '{"matches": ["strict", "none", "strict", ...]}',
        f"The array MUST have exactly {len(batch)} elements, in the same order.",
        "No explanations, no markdown fences, just the JSON.",
        "",
    ]
    for i, item in enumerate(batch):
        gt_str = " | ".join(item["ground_truths"][:8])
        tokens_str = ", ".join(f'"{t}"' for t in item["top_tokens_or_words"])
        lines.append(f"--- Item {i+1} ---")
        lines.append(f"Accepted answers: {gt_str}")
        lines.append(f"Top tokens: [{tokens_str}]")
        lines.append("")

    return "\n".join(lines)


# Valid match levels
MATCH_LEVELS = ("strict", "none")


def _parse_matches(raw_text: str, expected_len: int) -> Optional[List[Optional[bool]]]:
    """Parse the token-match judge's JSON response into a list of booleans (True=strict, False=none)."""
    if not raw_text:
        return None

    cleaned = re.sub(r"^(```json|```|''')\s*", "", raw_text.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        print(f"    [token-judge] WARNING: No JSON found: {raw_text[:200]}")
        return None

    try:
        data = json.loads(m.group(0))
        arr = data.get("matches", None)
        if not isinstance(arr, list):
            print(f"    [token-judge] WARNING: No 'matches' list: {data}")
            return None

        out: List[Optional[bool]] = []
        for x in arr[:expected_len]:
            if isinstance(x, str):
                xl = x.strip().lower()
                if xl in ("strict", "true", "yes"):
                    out.append(True)
                elif xl in ("none", "false", "no"):
                    out.append(False)
                else:
                    out.append(None)
            elif isinstance(x, bool):
                out.append(x)
            else:
                out.append(None)

        while len(out) < expected_len:
            out.append(None)
        return out

    except Exception as e:
        print(f"    [token-judge] WARNING: JSON parse error: {e}")
        return None


def judge_token_matches_batch(
    client,
    items: List[Dict],
    batch_size: int = JUDGE_BATCH_SIZE,
) -> List[Optional[bool]]:
    """
    Call the LLM to judge whether top-k tokens match the ground truth.

    Returns a list of booleans: True=strict match, False=no match, None=parse error.
    """
    all_matches: List[Optional[bool]] = []
    total = len(items)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = items[start:end]
        batch_num = start // batch_size + 1
        total_batches = math.ceil(total / batch_size)

        print(f"    [token-judge] Batch {batch_num}/{total_batches} (items {start+1}-{end}/{total})")

        prompt = _build_token_match_prompt(batch)

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=JUDGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=JUDGE_TEMPERATURE,
                        max_output_tokens=JUDGE_MAX_TOKENS,
                        candidate_count=1,
                    ),
                )
                raw = (resp.text or "").strip()
                parsed = _parse_matches(raw, expected_len=len(batch))
                if parsed is not None:
                    break
                print(f"    [token-judge] Parse failed on attempt {attempt}, retrying...")
            except Exception as e:
                print(f"    [token-judge] API error on attempt {attempt}: {e}")
                time.sleep(min(2.0 * attempt, 10.0))

        if parsed is None:
            parsed = [None] * len(batch)

        all_matches.extend(parsed)

    return all_matches


def rejudge_hit_rates(client, results: List[Dict]):
    """
    For each method and each result, ask the LLM whether the top-k
    tokens strictly match the ground truth (orthographic match).
    Stores results as judge_hit_<method_key> (bool) in each result dict.
    """
    methods = [
        ("Token perplexity", "perplexity_importance", "top_tokens", "top_tokens_text"),
        ("Token cosine",     "cosine_importance",     "top_tokens", "top_tokens_text"),
        ("Word perplexity",  "word_perplexity_importance", "top_words", "top_words_text"),
        ("Word cosine",      "word_cosine_importance",     "top_words", "top_words_text"),
    ]

    for label, key, top_key, text_key in methods:
        print(f"\n  [token-judge] Judging top-k for: {label}")

        items = []
        indices = []
        for i, r in enumerate(results):
            if key not in r:
                continue
            top_text = r[key]["comparison"].get(text_key, [])
            if not top_text:
                continue
            items.append({
                "question": r["question"],
                "ground_truths": r["ground_truths"],
                "top_tokens_or_words": top_text,
            })
            indices.append(i)

        if not items:
            print(f"    No items to judge for {label}")
            continue

        matches = judge_token_matches_batch(client, items, batch_size=JUDGE_BATCH_SIZE)

        judge_key = f"judge_hit_{key}"
        for idx, match in zip(indices, matches):
            results[idx][judge_key] = match

        n_match = sum(1 for m in matches if m is True)
        n_none = sum(1 for m in matches if m is False)
        print(f"    {label}: match={n_match}  none={n_none}  (total={len(matches)})")


# ──────────────────── Comparison analysis ───────────────────────

def compare_methods(results: List[Dict], figs_dir: Path):
    """
    Compare importance methods between judge-correct and judge-incorrect answers.
    Prints tables and generates figures.
    """
    correct = [r for r in results if r.get("judge_correct") is True]
    incorrect = [r for r in results if r.get("judge_correct") is False]
    unknown = [r for r in results if r.get("judge_correct") is None]

    n = len(results)
    nc = len(correct)
    ni = len(incorrect)
    nu = len(unknown)

    print(f"\n{'=' * 70}")
    print(f" JUDGE RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total:     {n}")
    print(f"  Correct:   {nc} ({nc/n:.1%})" if n > 0 else "  Correct:   0")
    print(f"  Incorrect: {ni} ({ni/n:.1%})" if n > 0 else "  Incorrect: 0")
    print(f"  Unknown:   {nu}")

    # Compare with substring matching
    substr_correct = sum(1 for r in results if r.get("answer_correct"))
    substr_agree = sum(1 for r in results
                       if r.get("judge_correct") is not None
                       and r.get("judge_correct") == r.get("answer_correct"))
    judged = nc + ni
    print(f"\n  Substring matching: {substr_correct}/{n} correct")
    if judged > 0:
        print(f"  Agreement judge vs substring: {substr_agree}/{judged} ({substr_agree/judged:.1%})")

    # ---- Methods config ----
    methods = [
        ("Token perplexity", "perplexity_importance"),
        ("Token cosine",     "cosine_importance"),
        ("Word perplexity",  "word_perplexity_importance"),
        ("Word cosine",      "word_cosine_importance"),
    ]

    # ---- Hit rates: correct vs incorrect ----
    print(f"\n{'=' * 70}")
    print(f" HIT RATES: ground truth in top-k important tokens/words")
    print(f"{'=' * 70}")
    print(f"  {'Method':<22s}  {'Substr C':>9s}  {'Substr I':>9s}  {'Judge C':>9s}  {'Judge I':>9s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    hit_data = {}  # label -> {"substr_c", "substr_i", "judge_c", "judge_i"}
    for label, key in methods:
        judge_key = f"judge_hit_{key}"
        comp_key = "answer_in_top_k"

        # Substring hit rate
        def _substr_rate(subset, k=key):
            valid = [r for r in subset if k in r]
            if not valid:
                return 0.0
            return sum(1 for r in valid if r[k]["comparison"][comp_key]) / len(valid)

        # LLM judge hit rate
        def _judge_rate(subset, jk=judge_key):
            judged = [r for r in subset if r.get(jk) is not None]
            if not judged:
                return 0.0
            return sum(1 for r in judged if r.get(jk) is True) / len(judged)

        sc = _substr_rate(correct)
        si = _substr_rate(incorrect)
        jc = _judge_rate(correct)
        ji = _judge_rate(incorrect)
        hit_data[label] = {"substr_c": sc, "substr_i": si, "judge_c": jc, "judge_i": ji}
        print(f"  {label:<22s}  {sc:>8.0%}  {si:>8.0%}  {jc:>8.0%}  {ji:>8.0%}")


    # ---- Perplexity stats ----
    print(f"\n{'=' * 70}")
    print(f" SENTENCE PERPLEXITY")
    print(f"{'=' * 70}")

    perp_correct = [r["sentence_perplexity"] for r in correct]
    perp_incorrect = [r["sentence_perplexity"] for r in incorrect]

    if perp_correct:
        print(f"  Correct answers:   mean={np.mean(perp_correct):.4f}  median={np.median(perp_correct):.4f}  std={np.std(perp_correct):.4f}")
    if perp_incorrect:
        print(f"  Incorrect answers: mean={np.mean(perp_incorrect):.4f}  median={np.median(perp_incorrect):.4f}  std={np.std(perp_incorrect):.4f}")

    # ---- Avg sentence perplexity: correct vs incorrect ----
    # (higher perplexity on incorrect answers = uncertainty detects errors)
    print(f"\n{'=' * 70}")
    print(f" AVG TOP-1 IMPORTANCE (normalized per method for comparison)")
    print(f"{'=' * 70}")
    print(f"  {'Method':<25s}  {'Correct':>12s}  {'Incorrect':>12s}  {'Ratio I/C':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}")

    importance_data = {}
    for label, key in methods:
        def _get_top1_values(subset, k=key):
            vals = []
            for r in subset:
                if k not in r:
                    continue
                top = r[k].get("top_tokens", r[k].get("top_words", []))
                if top:
                    vals.append(top[0]["importance"])
            return vals

        vals_c = _get_top1_values(correct)
        vals_i = _get_top1_values(incorrect)

        # Normalize: z-score over all values for this method
        all_vals = vals_c + vals_i
        if len(all_vals) >= 2:
            mu = np.mean(all_vals)
            sigma = np.std(all_vals)
            if sigma > 0:
                norm_c = [(v - mu) / sigma for v in vals_c]
                norm_i = [(v - mu) / sigma for v in vals_i]
            else:
                norm_c = [0.0] * len(vals_c)
                norm_i = [0.0] * len(vals_i)
        else:
            norm_c = vals_c
            norm_i = vals_i

        avg_c = np.mean(norm_c) if norm_c else float("nan")
        avg_i = np.mean(norm_i) if norm_i else float("nan")
        ratio = avg_i / avg_c if avg_c != 0 and not np.isnan(avg_c) else float("nan")

        importance_data[label] = {"correct": norm_c, "incorrect": norm_i}

        print(f"  {label:<25s}  {avg_c:>12.4f}  {avg_i:>12.4f}  {ratio:>10.2f}")

    # ---- Figures ----
    figs_dir.mkdir(exist_ok=True)

    _plot_hit_rate_bars(hit_data, figs_dir)
    _plot_perplexity_boxplot(perp_correct, perp_incorrect, figs_dir)
    _plot_importance_boxplots(importance_data, figs_dir)

    # ---- Incorrect answers with a top-k hit (LLM-judged) ----
    print(f"\n{'=' * 70}")
    print(f" INCORRECT ANSWERS WITH GROUND TRUTH IN TOP-K (LLM-judged)")
    print(f"{'=' * 70}")

    found_any = False
    for r in incorrect:
        hits = []
        for label, key in methods:
            judge_key = f"judge_hit_{key}"
            if r.get(judge_key) is True:
                top_text = (r[key]["comparison"].get("top_tokens_text")
                            or r[key]["comparison"].get("top_words_text", []))
                hits.append((label, top_text))
        if hits:
            found_any = True
            print(f"\n  Q: {r['question']}")
            print(f"  LLM answer:    {r['full_answer']}")
            print(f"  Ground truths: {r['ground_truths'][:5]}")
            for label, top_text in hits:
                print(f"    {label:<20s}  top-k={top_text}")

    if not found_any:
        print("  None found.")

    # ---- Per-question detail ----
    print(f"\n{'=' * 70}")
    print(f" DETAILED EXAMPLES (disagreements between judge and substring)")
    print(f"{'=' * 70}")
    disagreements = [r for r in results
                     if r.get("judge_correct") is not None
                     and r.get("judge_correct") != r.get("answer_correct")]
    if not disagreements:
        print("  No disagreements found.")
    for r in disagreements[:10]:
        print(f"\n  Q: {r['question']}")
        print(f"  LLM answer:    {r['full_answer']}")
        print(f"  Ground truths: {r['ground_truths'][:3]}")
        print(f"  Substring:     {'CORRECT' if r['answer_correct'] else 'INCORRECT'}")
        print(f"  Judge:         {'CORRECT' if r['judge_correct'] else 'INCORRECT'}")


# ──────────────────── Figures ───────────────────────────────────

def _plot_hit_rate_bars(hit_data: Dict, figs_dir: Path):
    """Two side-by-side charts: substring vs LLM-judge hit rates, one for correct, one for incorrect."""
    methods = list(hit_data.keys())
    x = np.arange(len(methods))
    width = 0.3

    sc = [hit_data[m]["substr_c"] for m in methods]
    si = [hit_data[m]["substr_i"] for m in methods]
    jc = [hit_data[m]["judge_c"] for m in methods]
    ji = [hit_data[m]["judge_i"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- Correct answers ---
    ax1.bar(x - width/2, sc, width, label="Substring", color="#4CAF50", alpha=0.85)
    ax1.bar(x + width/2, jc, width, label="LLM judge", color="#2196F3", alpha=0.85)
    for i in range(len(methods)):
        if sc[i] > 0:
            ax1.text(x[i] - width/2, sc[i] + 0.01, f"{sc[i]:.0%}", ha="center", va="bottom", fontsize=9)
        if jc[i] > 0:
            ax1.text(x[i] + width/2, jc[i] + 0.01, f"{jc[i]:.0%}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Hit rate (answer in top-k)")
    ax1.set_title("Correct answers")
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")

    # --- Incorrect answers ---
    ax2.bar(x - width/2, si, width, label="Substring", color="#F44336", alpha=0.85)
    ax2.bar(x + width/2, ji, width, label="LLM judge", color="#FF9800", alpha=0.85)
    for i in range(len(methods)):
        if si[i] > 0:
            ax2.text(x[i] - width/2, si[i] + 0.01, f"{si[i]:.0%}", ha="center", va="bottom", fontsize=9)
        if ji[i] > 0:
            ax2.text(x[i] + width/2, ji[i] + 0.01, f"{ji[i]:.0%}", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Incorrect answers")
    ax2.legend(fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha="right")

    fig.suptitle("Hit rate: substring matching vs LLM judge", fontsize=13, y=1.02)
    fig.tight_layout()
    path = figs_dir / "judge_hit_rate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_perplexity_boxplot(perp_correct: List, perp_incorrect: List, figs_dir: Path):
    """Boxplot: sentence perplexity for correct vs incorrect (judge)."""
    if not perp_correct and not perp_incorrect:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    colors = []
    if perp_correct:
        data.append(perp_correct)
        labels.append(f"Correct (n={len(perp_correct)})")
        colors.append("#4CAF50")
    if perp_incorrect:
        data.append(perp_incorrect)
        labels.append(f"Incorrect (n={len(perp_incorrect)})")
        colors.append("#F44336")

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Sentence perplexity")
    ax.set_title("Sentence perplexity: correct vs incorrect (LLM judge)")
    fig.tight_layout()
    path = figs_dir / "judge_perplexity_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_importance_boxplots(importance_data: Dict, figs_dir: Path):
    """Boxplots: normalized top-1 importance, correct vs incorrect, for each method."""
    methods = list(importance_data.keys())
    n_methods = len(methods)

    if n_methods == 0:
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        data_c = importance_data[method]["correct"]
        data_i = importance_data[method]["incorrect"]

        box_data = []
        box_labels = []
        box_colors = []
        if data_c:
            box_data.append(data_c)
            box_labels.append(f"Correct\n(n={len(data_c)})")
            box_colors.append("#4CAF50")
        if data_i:
            box_data.append(data_i)
            box_labels.append(f"Incorrect\n(n={len(data_i)})")
            box_colors.append("#F44336")

        if not box_data:
            continue

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(method, fontsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    axes[0].set_ylabel("Normalized top-1 importance (z-score)")
    fig.suptitle("Top-1 importance: correct vs incorrect (LLM judge)", fontsize=12, y=1.02)
    fig.tight_layout()
    path = figs_dir / "judge_importance_boxplots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ──────────────────────────── Main ──────────────────────────────

def main():
    load_dotenv()

    # ---- Vertex AI client ----
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")
    client = genai.Client(vertexai=True, project=project, location=location)

    # ---- Load results ----
    if not RESULTS_JSONL.exists():
        print(f"Results file not found: {RESULTS_JSONL}")
        print("Run main_triviaqa.py first.")
        return

    results = load_results(RESULTS_JSONL)
    print(f"Loaded {len(results)} valid results from {RESULTS_JSONL.name}")

    if not results:
        print("No results to process.")
        return

    # ---- LLM-as-judge ----
    print(f"\n{'=' * 70}")
    print(f" LLM-AS-JUDGE (model={JUDGE_MODEL}, batch_size={JUDGE_BATCH_SIZE})")
    print(f"{'=' * 70}")

    judgments = judge_answers_batch(client, results, batch_size=JUDGE_BATCH_SIZE)

    # Enrich results with judge verdicts
    for r, j in zip(results, judgments):
        r["judge_correct"] = j

    # ---- LLM-judge token matching ----
    print(f"\n{'=' * 70}")
    print(f" LLM-AS-JUDGE FOR TOP-K TOKEN MATCHING")
    print(f"{'=' * 70}")
    rejudge_hit_rates(client, results)

    # ---- Save judged results ----
    with open(JUDGED_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved judged results to {JUDGED_JSONL.name}")

    # ---- Compare methods ----
    compare_methods(results, FIGS_DIR)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
