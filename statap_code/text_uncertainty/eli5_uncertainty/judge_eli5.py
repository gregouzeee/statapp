"""
ELI5 – LLM-as-judge evaluation (batched, scale 1-5)
=====================================================
1.  Load existing results from eli5_results.jsonl
2.  Use Gemini as a judge to score each answer on a 1-5 scale
    across 3 criteria: simplicity, accuracy, relevance.
3.  Save enriched results to eli5_judged.jsonl
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ──────────────────────────── Config ────────────────────────────

JUDGE_MODEL = "gemini-2.5-flash"
JUDGE_MAX_TOKENS = 8192  # 2.5 Flash uses tokens for internal thinking
JUDGE_BATCH_SIZE = 10
MAX_RETRIES = 3

RESULTS_JSONL = Path(__file__).resolve().parent / "eli5_results.jsonl"
JUDGED_JSONL  = Path(__file__).resolve().parent / "eli5_judged.jsonl"
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

def _build_eli5_judge_prompt(batch: List[Dict]) -> str:
    """
    Build a batched prompt for the ELI5 judge.
    Each answer is scored 1-5 on three criteria.
    """
    lines = [
        "You are an expert evaluator of 'Explain Like I'm 5' (ELI5) answers.",
        "",
        "Score each answer on THREE criteria, each on a scale from 1 to 5:",
        "",
        "1. SIMPLICITY (1-5):",
        "   1 = Very technical, jargon-heavy, incomprehensible for a child",
        "   2 = Mostly technical with some simple parts",
        "   3 = Mixed: some simple explanations but still uses complex words",
        "   4 = Mostly simple, occasional slightly advanced word",
        "   5 = Perfect ELI5: simple words, short sentences, great analogies",
        "",
        "2. ACCURACY (1-5):",
        "   1 = Mostly wrong or severely misleading",
        "   2 = Contains significant factual errors",
        "   3 = Roughly correct but imprecise or missing key nuances",
        "   4 = Correct with minor simplifications (acceptable for ELI5)",
        "   5 = Factually accurate, well-aligned with the reference answer",
        "",
        "3. RELEVANCE (1-5):",
        "   1 = Does not address the question at all",
        "   2 = Tangentially related but misses the point",
        "   3 = Partially addresses the question",
        "   4 = Addresses the question but misses a secondary aspect",
        "   5 = Directly and fully addresses the question",
        "",
        "Use the reference answer (from Reddit) as a guide for accuracy,",
        "but do not require an exact match.",
        "",
        "Return ONLY a JSON object in this exact format:",
        '{"scores": [{"simplicity": 4, "accuracy": 5, "relevance": 3}, ...]}',
        f"The array MUST have exactly {len(batch)} elements, in the same order.",
        "Each score must be an integer from 1 to 5.",
        "No explanations, no markdown fences, just the JSON.",
        "",
    ]
    for i, item in enumerate(batch):
        ref = item.get("reference_answer", "")
        ref_short = ref[:300] + "..." if len(ref) > 300 else ref
        lines.append(f"--- Item {i+1} ---")
        lines.append(f"Question: {item['question']}")
        lines.append(f"Reference answer (Reddit): {ref_short}")
        lines.append(f"LLM answer: {item['full_answer']}")
        lines.append("")

    return "\n".join(lines)


def _parse_scores(raw_text: str, expected_len: int) -> Optional[List[Optional[Dict]]]:
    """Parse the judge's JSON response into a list of score dicts."""
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
        arr = data.get("scores", None)
        if not isinstance(arr, list):
            print(f"    [judge] WARNING: No 'scores' list in JSON: {data}")
            return None

        out: List[Optional[Dict]] = []
        for x in arr[:expected_len]:
            if isinstance(x, dict):
                score = {}
                for key in ("simplicity", "accuracy", "relevance"):
                    val = x.get(key)
                    if isinstance(val, (int, float)) and 1 <= val <= 5:
                        score[key] = int(val)
                    else:
                        score[key] = None
                out.append(score)
            else:
                out.append(None)

        # Pad if too short
        while len(out) < expected_len:
            out.append(None)
        return out

    except Exception as e:
        print(f"    [judge] WARNING: JSON parse error: {e}")
        return None


def judge_eli5_batch(
    client,
    results: List[Dict],
    batch_size: int = JUDGE_BATCH_SIZE,
) -> List[Optional[Dict]]:
    """
    Call the LLM judge on all results in batches.
    Returns a list of score dicts: {"simplicity": int, "accuracy": int, "relevance": int}
    or None on parse error.
    """
    all_scores: List[Optional[Dict]] = []
    total = len(results)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = results[start:end]
        batch_num = start // batch_size + 1
        total_batches = math.ceil(total / batch_size)

        print(f"\n  [judge] Batch {batch_num}/{total_batches} "
              f"(items {start+1}-{end}/{total})")

        prompt = _build_eli5_judge_prompt(batch)

        parsed = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=JUDGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=JUDGE_MAX_TOKENS,
                        candidate_count=1,
                    ),
                )
                raw = (resp.text or "").strip()
                parsed = _parse_scores(raw, expected_len=len(batch))
                if parsed is not None:
                    break
                print(f"    [judge] Parse failed on attempt {attempt}, retrying...")
            except Exception as e:
                print(f"    [judge] API error on attempt {attempt}: {e}")
                time.sleep(min(2.0 * attempt, 10.0))

        if parsed is None:
            print("    [judge] All retries failed — defaulting to None")
            parsed = [None] * len(batch)

        for i, (item, score) in enumerate(zip(batch, parsed)):
            if score:
                s = score.get("simplicity", "?")
                a = score.get("accuracy", "?")
                r = score.get("relevance", "?")
                print(f"    [{start+i+1}] S={s} A={a} R={r}  "
                      f"Q: {item['question'][:55]}...")
            else:
                print(f"    [{start+i+1}] PARSE ERROR  "
                      f"Q: {item['question'][:55]}...")

        all_scores.extend(parsed)

    return all_scores


# ──────────────────────────── Main ──────────────────────────────

def main():
    load_dotenv()

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    # ---- Load results ----
    results = load_results(RESULTS_JSONL)
    print(f"Loaded {len(results)} valid results from {RESULTS_JSONL}")

    if not results:
        print("No results to judge.")
        return

    # ---- Judge ----
    print("\n" + "=" * 70)
    print("JUDGING ELI5 ANSWERS (scale 1-5)")
    print("=" * 70)

    scores = judge_eli5_batch(client, results)
    for r, s in zip(results, scores):
        r["judge_scores"] = s

    # ---- Save ----
    with open(JUDGED_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} judged results to {JUDGED_JSONL}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_scores = [r["judge_scores"] for r in results
                    if r.get("judge_scores") is not None]
    n = len(results)
    n_valid = len(valid_scores)
    n_failed = n - n_valid

    print(f"  Total:         {n}")
    print(f"  Scored:        {n_valid}")
    print(f"  Parse errors:  {n_failed}")

    if valid_scores:
        for key in ("simplicity", "accuracy", "relevance"):
            vals = [s[key] for s in valid_scores if s.get(key) is not None]
            if vals:
                print(f"\n  {key.upper()}:")
                print(f"    mean={np.mean(vals):.2f}  median={np.median(vals):.1f}  "
                      f"min={min(vals)}  max={max(vals)}")
                # Distribution
                for score in range(1, 6):
                    count = vals.count(score)
                    bar = "#" * count
                    print(f"    {score}: {count:3d}  {bar}")

    # ---- Compare uncertainty metrics by score ----
    scored_results = [r for r in results if r.get("judge_scores") is not None]
    if scored_results:
        # Average score = mean(simplicity, accuracy, relevance)
        for r in scored_results:
            s = r["judge_scores"]
            vals = [v for v in [s.get("simplicity"), s.get("accuracy"),
                                s.get("relevance")] if v is not None]
            r["judge_avg_score"] = np.mean(vals) if vals else None

        # Split into high (>=4) vs low (<4)
        high = [r for r in scored_results
                if r.get("judge_avg_score") is not None and r["judge_avg_score"] >= 4]
        low = [r for r in scored_results
               if r.get("judge_avg_score") is not None and r["judge_avg_score"] < 4]

        if high and low:
            print(f"\n  Uncertainty metrics: high-quality (avg>=4, n={len(high)}) "
                  f"vs low-quality (avg<4, n={len(low)})")

            # Per-sentence perplexity
            high_ppls = [s["perplexity"] for r in high for s in r["sentences"]]
            low_ppls = [s["perplexity"] for r in low for s in r["sentences"]]
            print(f"    Per-sentence perplexity:")
            print(f"      High-quality: mean={np.mean(high_ppls):.2f}  "
                  f"median={np.median(high_ppls):.2f}")
            print(f"      Low-quality:  mean={np.mean(low_ppls):.2f}  "
                  f"median={np.median(low_ppls):.2f}")

            # Per-sentence top-k entropy
            high_h = [s["topk_entropy"] for r in high for s in r["sentences"]]
            low_h = [s["topk_entropy"] for r in low for s in r["sentences"]]
            print(f"    Per-sentence top-k entropy:")
            print(f"      High-quality: mean={np.mean(high_h):.3f}  "
                  f"median={np.median(high_h):.3f}")
            print(f"      Low-quality:  mean={np.mean(low_h):.3f}  "
                  f"median={np.median(low_h):.3f}")

            # Overall perplexity
            high_ov = [r["overall_perplexity"] for r in high]
            low_ov = [r["overall_perplexity"] for r in low]
            print(f"    Overall perplexity:")
            print(f"      High-quality: mean={np.mean(high_ov):.2f}  "
                  f"median={np.median(high_ov):.2f}")
            print(f"      Low-quality:  mean={np.mean(low_ov):.2f}  "
                  f"median={np.median(low_ov):.2f}")

    print()


if __name__ == "__main__":
    main()
