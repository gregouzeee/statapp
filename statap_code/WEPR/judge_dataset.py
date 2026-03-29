"""
WEPR Dataset Annotation — LLM-as-a-Judge
=========================================
Annotates the generated TriviaQA answers as correct (Y=1) or
hallucinated (Y=0) using Gemini as a judge.

Reads: data/triviaqa_topk.jsonl
Writes: data/triviaqa_judged.jsonl

Usage:
    python judge_dataset.py [--batch_size 20]
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


# ──────────────────────────── Config ────────────────────────────

DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 3

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_JSONL = DATA_DIR / "triviaqa_topk.jsonl"
OUTPUT_JSONL = DATA_DIR / "triviaqa_judged.jsonl"


# ──────────────────── Load data ─────────────────────

def load_results(path: Path) -> List[Dict]:
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


# ──────────────────── Judge prompt ────────────────────

def build_judge_prompt(batch: List[Dict]) -> str:
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


def parse_judgments(raw_text: str, expected_len: int) -> Optional[List[Optional[bool]]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"^(```json|```|''')\s*", "", raw_text.strip(),
                     flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        arr = data.get("judgments", None)
        if not isinstance(arr, list):
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
        while len(out) < expected_len:
            out.append(None)
        return out
    except Exception:
        return None


# ──────────────────── Main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Judge TriviaQA answers")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--input", type=str, default=str(INPUT_JSONL))
    parser.add_argument("--output", type=str, default=str(OUTPUT_JSONL))
    args = parser.parse_args()

    load_dotenv()

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")
    client = genai.Client(vertexai=True, project=project, location=location)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load results
    results = load_results(input_path)
    print(f"Loaded {len(results)} valid results from {input_path.name}")
    if not results:
        print("No results to judge.")
        return

    # Check already judged
    already_judged = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "question_id" in obj and "judge_correct" in obj:
                        already_judged.add(obj["question_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Already judged: {len(already_judged)}")

    to_judge = [r for r in results if r["question_id"] not in already_judged]
    print(f"To judge: {len(to_judge)}")

    if not to_judge:
        print("Nothing to do.")
        return

    # Judge in batches
    total = len(to_judge)
    batch_size = args.batch_size

    with open(output_path, "a", encoding="utf-8") as fout:
        for start in tqdm(range(0, total, batch_size), desc="Judging", unit="batch"):
            end = min(start + batch_size, total)
            batch = to_judge[start:end]

            prompt = build_judge_prompt(batch)
            parsed = None

            for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
                try:
                    resp = client.models.generate_content(
                        model=args.judge_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=256,
                            candidate_count=1,
                        ),
                    )
                    raw = (resp.text or "").strip()
                    parsed = parse_judgments(raw, expected_len=len(batch))
                    if parsed is not None:
                        break
                except Exception as e:
                    time.sleep(min(2.0 * attempt, 10.0))

            if parsed is None:
                parsed = [None] * len(batch)

            for item, judgment in zip(batch, parsed):
                item["judge_correct"] = judgment
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    # Summary
    correct = 0
    incorrect = 0
    unknown = 0
    total_judged = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            jc = obj.get("judge_correct")
            total_judged += 1
            if jc is True:
                correct += 1
            elif jc is False:
                incorrect += 1
            else:
                unknown += 1

    print(f"\nDone. Total: {total_judged}")
    print(f"  Correct:   {correct} ({correct/total_judged:.1%})")
    print(f"  Incorrect: {incorrect} ({incorrect/total_judged:.1%})")
    print(f"  Unknown:   {unknown}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
