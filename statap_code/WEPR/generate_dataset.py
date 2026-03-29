"""
WEPR Dataset Generation — TriviaQA with full top-K logprobs
===========================================================
Generates answers to TriviaQA questions using Gemini and saves
the full top-K log-probabilities at each token position.

This is the data needed to train and evaluate WEPR.

Usage:
    python generate_dataset.py [--num_questions 1000] [--logprobs_k 10] [--concurrency 10]
"""

import argparse
import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


# ──────────────────────────── Defaults ────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_LOGPROBS_K = 10
DEFAULT_MAX_OUTPUT_TOKENS = 128
DEFAULT_TEMPERATURE = 1.0  # T=1.0 comme dans l'article EPR
DEFAULT_NUM_QUESTIONS = 0  # 0 = all available questions
DEFAULT_MAX_RETRIES = 5
DEFAULT_CONCURRENCY = 10

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_JSONL = OUT_DIR / "triviaqa_topk.jsonl"


# ──────────────────────────── Prompt ────────────────────────────

def make_prompt(question: str) -> str:
    return f"""Answer the following question in exactly ONE short sentence.
Your sentence must contain the answer.
Do not add any preamble, explanation, or follow-up.

Question: {question}
Answer:""".strip()


# ──────────── Log-prob extraction (full top-K) ───────────────

def extract_token_logprobs(resp) -> List[Dict]:
    """
    Extract per-token info with full top-K candidates:
      [{ "token": str, "log_prob": float,
         "top_k": [{"token": str, "log_prob": float}, ...] }, ...]
    """
    tokens = []
    for cand in getattr(resp, "candidates", []) or []:
        lpr = getattr(cand, "logprobs_result", None)
        if not lpr:
            continue
        chosen = getattr(lpr, "chosen_candidates", []) or []
        top_cands = getattr(lpr, "top_candidates", []) or []
        for i, cc in enumerate(chosen):
            entry = {
                "token": cc.token,
                "log_prob": float(cc.log_probability),
                "top_k": [],
            }
            if i < len(top_cands):
                for tc in (top_cands[i].candidates or []):
                    entry["top_k"].append({
                        "token": tc.token,
                        "log_prob": float(tc.log_probability),
                    })
            tokens.append(entry)
    return tokens


# ──────────── Single question processing ───────────────

def process_one(client, item, args) -> dict:
    """Process a single question: call Gemini, extract logprobs, return result dict."""
    qid = item["question_id"]
    question = item["question"]
    answer_obj = item["answer"]
    ground_truths = []
    if answer_obj.get("value"):
        ground_truths.append(answer_obj["value"])
    ground_truths.extend(answer_obj.get("aliases", []))
    ground_truths.extend(answer_obj.get("normalized_aliases", []))
    ground_truths = list(dict.fromkeys(ground_truths))

    prompt = make_prompt(question)

    resp = None
    last_err = None
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=args.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    response_logprobs=True,
                    logprobs=args.logprobs_k,
                    candidate_count=1,
                ),
            )
            break
        except Exception as e:
            last_err = str(e)
            if any(fatal in last_err for fatal in
                   ["Credentials", "credentials", "BILLING",
                    "PERMISSION_DENIED", "limit: 0"]):
                raise
            # Exponential backoff, longer on rate limit
            if "429" in last_err or "RESOURCE_EXHAUSTED" in last_err:
                time.sleep(min(5.0 * attempt, 30.0))
            else:
                time.sleep(min(2.0 * attempt, 10.0))

    if resp is None:
        return {
            "question_id": qid,
            "question": question,
            "ground_truths": ground_truths,
            "error": last_err,
            "model": args.model,
        }

    full_answer = (resp.text or "").strip()
    token_data = extract_token_logprobs(resp)

    if not token_data:
        return {
            "question_id": qid,
            "question": question,
            "ground_truths": ground_truths,
            "full_answer": full_answer,
            "error": "no logprobs returned",
            "model": args.model,
        }

    return {
        "question_id": qid,
        "question": question,
        "ground_truths": ground_truths,
        "full_answer": full_answer,
        "model": args.model,
        "logprobs_k": args.logprobs_k,
        "temperature": args.temperature,
        "token_data": token_data,
    }


# ──────────────────────────── Main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate TriviaQA dataset with top-K logprobs")
    parser.add_argument("--num_questions", type=int, default=DEFAULT_NUM_QUESTIONS,
                        help="Number of questions (0 = all available)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["validation", "train"],
                        help="TriviaQA split to use")
    parser.add_argument("--logprobs_k", type=int, default=DEFAULT_LOGPROBS_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of parallel API calls")
    args = parser.parse_args()

    load_dotenv()

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    # Load TriviaQA (streaming — never loads everything in RAM)
    from datasets import load_dataset

    print(f"Loading TriviaQA (split={args.split}, streaming)...")
    ds_stream = load_dataset("trivia_qa", "rc", split=args.split, streaming=True)
    if args.num_questions > 0:
        ds = ds_stream.take(args.num_questions)
        total = args.num_questions
        print(f"Will process up to {args.num_questions} questions (streaming).")
    else:
        ds = ds_stream
        total = None
        print("Will process all available questions (streaming).")

    # Resume logic
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and OUT_JSONL.exists():
        with open(OUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "question_id" in obj:
                        done.add(obj["question_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Already processed: {len(done)} questions. Resuming...")

    print(f"Concurrency: {args.concurrency} parallel requests")

    # File write lock (threads write results concurrently)
    write_lock = threading.Lock()

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_async(item, fout, pbar):
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_one, client, item, args)
            line = json.dumps(result, ensure_ascii=False) + "\n"
            with write_lock:
                fout.write(line)
                fout.flush()
            pbar.update(1)

    async def run_all():
        with open(OUT_JSONL, "a", encoding="utf-8") as fout:
            pbar = tqdm(desc="Generating", unit="q", total=total)
            tasks = []
            for item in ds:
                qid = item["question_id"]
                if args.resume and qid in done:
                    pbar.update(1)
                    continue
                tasks.append(process_async(item, fout, pbar))
                # Launch in batches to avoid buffering too many items from the stream
                if len(tasks) >= args.concurrency * 2:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)
            pbar.close()

    asyncio.run(run_all())

    # Count results
    total_count = 0
    errors = 0
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_count += 1
            if "error" in obj:
                errors += 1
    print(f"\nDone. {total_count} entries ({total_count - errors} valid, {errors} errors)")
    print(f"Output: {OUT_JSONL}")


if __name__ == "__main__":
    main()
