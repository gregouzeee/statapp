"""
WEPR Dataset Annotation — LLM-as-a-Judge
=========================================
Annotates the generated TriviaQA answers as correct (Y=1) or
hallucinated (Y=0) using Gemini as a judge.

Reads: data/triviaqa_topk.jsonl
Writes: data/triviaqa_judged.jsonl

Usage:
    python judge_dataset.py [--batch_size 20] [--concurrency 5]
"""

import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# Set root logger to ERROR so all libraries are silent by default
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("judge")
log.setLevel(logging.INFO)


# ──────────────────────────── Config ────────────────────────────

DEFAULT_JUDGE_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 5
DEFAULT_CONCURRENCY = 5

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


# ──────────── Process one batch ───────────────

def judge_one_batch(client, batch: List[Dict], args) -> List[Optional[bool]]:
    """Judge a single batch, return list of judgments."""
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
            err = str(e)
            if any(fatal in err for fatal in
                   ["Credentials", "credentials", "BILLING",
                    "PERMISSION_DENIED"]):
                log.error(f"FATAL: {err}")
                raise
            is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
            if is_rate_limit:
                wait = min(5.0 * attempt, 30.0)
                log.warning(f"429 RATE LIMITED (attempt {attempt}/{DEFAULT_MAX_RETRIES}). "
                            f"Waiting {wait:.0f}s...")
            else:
                wait = min(2.0 * attempt, 10.0)
                log.warning(f"ERROR (attempt {attempt}/{DEFAULT_MAX_RETRIES}): "
                            f"{err[:150]}. Waiting {wait:.0f}s...")
            time.sleep(wait)

    if parsed is None:
        return [None] * len(batch)
    return parsed


# ──────────────────── Main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Judge TriviaQA answers")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--input", type=str, default=str(INPUT_JSONL))
    parser.add_argument("--output", type=str, default=str(OUTPUT_JSONL))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of parallel judge calls (default: 5)")
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

    # Split into batches
    batch_size = args.batch_size
    batches = []
    for start in range(0, len(to_judge), batch_size):
        batches.append(to_judge[start:start + batch_size])

    print(f"Concurrency: {args.concurrency} parallel calls, {len(batches)} batches of {batch_size}")

    # Shared state
    write_lock = threading.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def judge_async(batch, fout, pbar):
        async with semaphore:
            loop = asyncio.get_event_loop()
            judgments = await loop.run_in_executor(
                None, judge_one_batch, client, batch, args
            )
            with write_lock:
                for item, judgment in zip(batch, judgments):
                    item["judge_correct"] = judgment
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
            pbar.update(1)

    async def run_all():
        t_start = time.monotonic()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a", encoding="utf-8") as fout:
            pbar = tqdm(desc="Judging", unit="batch", total=len(batches))
            tasks = []
            for batch in batches:
                tasks.append(judge_async(batch, fout, pbar))
                if len(tasks) >= args.concurrency * 2:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)
            pbar.close()

        elapsed = time.monotonic() - t_start
        log.info(f"Judging done in {elapsed:.0f}s")

    asyncio.run(run_all())

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
