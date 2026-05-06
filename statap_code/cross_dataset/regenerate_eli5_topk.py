"""
Regenerate ELI5 long-form answers with top-K log-probabilities
==============================================================

`eli5_judged.jsonl` stores `token_data` but only with the chosen
log_prob — not the per-position top-K. EPR / WEPR therefore cannot be
computed. This script re-runs the same prompt as `main_eli5.py` on the
same 100 ELI5 questions with `logprobs_k=10`, and stores the full
top-K so cross_table.py can compute EPR/WEPR on ELI5.

Output: data/eli5_topk.jsonl with one line per question, joinable to
`eli5_judged.jsonl` via `question_id`.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)
log = logging.getLogger("regen_eli5")
log.setLevel(logging.INFO)


# ──────────────────────────── Config ────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_LOGPROBS_K = 10
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_QUESTIONS = 100
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parent
SOURCE_JSONL = (ROOT.parent / "text_uncertainty" / "eli5_uncertainty"
                / "eli5_judged.jsonl")
OUT_JSONL = ROOT / "data" / "eli5_topk.jsonl"


def make_eli5_prompt(question: str) -> str:
    return (
        "Explain the following like I'm 5 years old.\n"
        "Use simple words, short sentences, and everyday analogies.\n"
        "Keep your explanation to 3-5 sentences maximum.\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Explanation:"
    )


def extract_token_logprobs(resp) -> List[Dict]:
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


class RateLimitTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.hits = 0

    def record_hit(self):
        with self.lock:
            self.hits += 1

    @property
    def total_hits(self):
        with self.lock:
            return self.hits


def process_one(client, item: Dict, args,
                rate_tracker: RateLimitTracker) -> Dict:
    qid = item.get("question_id")
    question = item["question"]
    prompt = make_eli5_prompt(question)

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
            if any(f in last_err for f in
                   ["Credentials", "credentials", "BILLING",
                    "PERMISSION_DENIED", "limit: 0"]):
                raise
            if "429" in last_err or "RESOURCE_EXHAUSTED" in last_err:
                rate_tracker.record_hit()
                wait = min(5.0 * attempt, 30.0)
            else:
                wait = min(2.0 * attempt, 10.0)
            time.sleep(wait)

    if resp is None:
        return {
            "question_id": qid,
            "question": question,
            "error": last_err,
            "model": args.model,
        }

    full_text = (resp.text or "").strip()
    token_data = extract_token_logprobs(resp)
    if not token_data:
        return {
            "question_id": qid,
            "question": question,
            "full_answer": full_text,
            "error": "no logprobs",
            "model": args.model,
        }

    return {
        "question_id": qid,
        "question": question,
        "full_answer": full_text,
        "model": args.model,
        "logprobs_k": args.logprobs_k,
        "temperature": args.temperature,
        "n_tokens": len(token_data),
        "token_data": token_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=DEFAULT_NUM_QUESTIONS)
    parser.add_argument("--logprobs_k", type=int, default=DEFAULT_LOGPROBS_K)
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_output_tokens", type=int,
                        default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--concurrency", type=int,
                        default=DEFAULT_CONCURRENCY)
    parser.add_argument("--input", type=str, default=str(SOURCE_JSONL))
    parser.add_argument("--output", type=str, default=str(OUT_JSONL))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    load_dotenv()
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")
    client = genai.Client(vertexai=True, project=project, location=location)

    src = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" not in obj or "question_id" not in obj:
                continue
            src.append({"question_id": obj["question_id"],
                        "question": obj["question"]})
    src = src[:args.num]
    log.info(f"Loaded {len(src)} ELI5 questions from {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "error" not in obj and "question_id" in obj:
                        done.add(obj["question_id"])
                except json.JSONDecodeError:
                    continue
    log.info(f"Already done: {len(done)} (resuming).")
    todo = [it for it in src if it["question_id"] not in done]
    log.info(f"To process: {len(todo)}")

    write_lock = threading.Lock()
    rate_tracker = RateLimitTracker()
    semaphore = asyncio.Semaphore(args.concurrency)

    counters = {"done": 0, "ok": 0, "err": 0,
                "tokens_total": 0, "t0": time.monotonic()}

    def report():
        elapsed = time.monotonic() - counters["t0"]
        rate = counters["done"] / elapsed if elapsed > 0 else 0.0
        eta = ((len(todo) - counters["done"]) / rate) if rate > 0 else float("inf")
        avg_tok = counters["tokens_total"] / max(1, counters["ok"])
        log.info(
            f"progress {counters['done']}/{len(todo)} "
            f"({100 * counters['done'] / max(1, len(todo)):.1f}%) "
            f"| ok={counters['ok']} err={counters['err']} "
            f"| avg_tok={avg_tok:.0f} | {rate:.2f} q/s "
            f"| ETA {eta / 60:.1f} min | 429s={rate_tracker.total_hits}"
        )

    async def process_async(item, fout, pbar):
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, process_one, client, item, args, rate_tracker
            )
            line = json.dumps(result, ensure_ascii=False) + "\n"
            with write_lock:
                fout.write(line)
                fout.flush()
                counters["done"] += 1
                if "error" in result:
                    counters["err"] += 1
                else:
                    counters["ok"] += 1
                    counters["tokens_total"] += result.get("n_tokens", 0)
                if counters["done"] % 10 == 0 or counters["done"] == len(todo):
                    report()
            pbar.update(1)

    async def run_all():
        with open(out_path, "a", encoding="utf-8") as fout:
            pbar = tqdm(total=len(todo), desc="ELI5 topK", unit="q",
                        file=sys.stderr, mininterval=2.0,
                        dynamic_ncols=True)
            tasks = [process_async(it, fout, pbar) for it in todo]
            chunk = args.concurrency * 4
            for i in range(0, len(tasks), chunk):
                await asyncio.gather(*tasks[i:i + chunk])
            pbar.close()

    log.info(f"Starting | model={args.model} K={args.logprobs_k} "
             f"max_tokens={args.max_output_tokens} T={args.temperature} "
             f"concurrency={args.concurrency}")
    t0 = time.monotonic()
    asyncio.run(run_all())
    log.info(f"Total: {time.monotonic() - t0:.0f}s "
             f"({rate_tracker.total_hits} rate limits)")


if __name__ == "__main__":
    main()
