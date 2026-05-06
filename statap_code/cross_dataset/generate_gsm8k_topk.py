"""
GSM8K — Generate long-form Chain-of-Thought answers with top-K logprobs
=======================================================================

Generates answers to GSM8K questions using Gemini, asking for a detailed
reasoning followed by a final numeric answer in JSON. We keep the full
top-K log-probabilities at every token position so that WEPR features
can be computed downstream.

Output format (compatible with WEPR/wepr.py and WEPR/train_wepr.py):

    {
      "question_id": str,
      "question": str,
      "ground_truth": float,        # gold numeric answer from GSM8K
      "full_answer": str,            # full text returned by the model
      "model": str,
      "logprobs_k": int,
      "temperature": float,
      "token_data": [
        {"token": str, "log_prob": float,
         "top_k": [{"token": str, "log_prob": float}, ...]}, ...
      ],
      "reasoning_token_data": [...],   # tokens of the reasoning sub-sequence
      "answer_token_data": [...],      # tokens of the final numeric answer
      "answer_text": str,              # raw answer text extracted from JSON
      "answer_value": float|None,      # parsed numeric value
      "n_tokens_total": int,
      "n_tokens_reasoning": int,
      "n_tokens_answer": int,
    }

Source: statap_code/logit_gsm8k/dataset_gsm8k.jsonl
Output: statap_code/cross_dataset/data/gsm8k_topk.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)
log = logging.getLogger("gsm8k_topk")
log.setLevel(logging.INFO)


# ──────────────────────────── Defaults ────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_LOGPROBS_K = 10
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NUM_QUESTIONS = 300
DEFAULT_MAX_RETRIES = 5
DEFAULT_CONCURRENCY = 6

ROOT = Path(__file__).resolve().parent
SOURCE_JSONL = ROOT.parent / "logit_gsm8k" / "dataset_gsm8k.jsonl"
OUT_JSONL = ROOT / "data" / "gsm8k_topk.jsonl"


# ──────────────────────────── Prompt ────────────────────────────

PROMPT_TEMPLATE = """You are a careful math problem solver.

Solve the following grade-school math problem. First, work through the
problem step by step in clear English. Then, on the LAST line, output a
JSON object with the final numeric answer in the form:

{{"answer": <number>}}

Rules:
- The reasoning must precede the JSON. Use plain English, no markdown.
- The final line must be ONLY the JSON object — no fences, no extra text.
- "answer" must be a number in decimal notation (no commas, no units, no text).

Question:
{q}
"""


def make_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(q=question.strip())


# ──────────── Log-prob extraction (full top-K) ───────────────

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


# ──────────── Reasoning vs answer split ──────────────────────

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_final_answer_text(full_text: str) -> Tuple[Optional[str],
                                                     Optional[float]]:
    """
    Find the last JSON-looking object in the text and parse 'answer'.
    Returns (raw_answer_text, numeric_value_or_None).
    """
    if not full_text:
        return None, None
    # Look for the last {...} block in the text
    matches = list(re.finditer(r"\{[^{}]*\"answer\"[^{}]*\}", full_text))
    if not matches:
        # fallback: any number in the last line
        last = full_text.strip().splitlines()[-1] if full_text.strip() else ""
        m = _NUM_RE.search(last)
        if m:
            try:
                return last, float(m.group(0))
            except ValueError:
                pass
        return None, None

    last_match = matches[-1]
    raw = last_match.group(0)
    try:
        obj = json.loads(raw)
        v = obj.get("answer")
        if isinstance(v, (int, float)):
            return raw, float(v)
        if isinstance(v, str):
            mm = _NUM_RE.search(v)
            if mm:
                return raw, float(mm.group(0))
    except Exception:
        pass
    # last-resort: parse a number out of `raw`
    mm = _NUM_RE.search(raw)
    if mm:
        try:
            return raw, float(mm.group(0))
        except ValueError:
            pass
    return raw, None


def split_reasoning_answer(token_data: List[Dict],
                            full_text: str,
                            answer_raw: Optional[str]
                            ) -> Tuple[List[Dict], List[Dict]]:
    """
    Split token_data into (reasoning_tokens, answer_tokens) by reconstructing
    the running text and locating the start of `answer_raw`.

    If anchor cannot be found, returns (token_data, []).
    """
    if not answer_raw or not token_data:
        return token_data, []

    # Find the byte index of answer_raw in full_text (last occurrence)
    try:
        idx = full_text.rindex(answer_raw)
    except ValueError:
        return token_data, []

    # Reconstruct cumulative text token by token; find first token whose
    # cumulative length is > idx -> first token in the answer.
    cum = 0
    split_at = None
    for k, td in enumerate(token_data):
        tk = td.get("token", "")
        cum_next = cum + len(tk)
        if cum < idx <= cum_next or cum >= idx:
            split_at = k
            break
        cum = cum_next

    if split_at is None:
        # answer is at the very end — fallback to last 6 tokens
        split_at = max(0, len(token_data) - 6)

    return token_data[:split_at], token_data[split_at:]


# ──────────── Rate limit tracker ───────────────

class RateLimitTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.hits = 0
        self.last_hit_time = 0.0

    def record_hit(self):
        with self.lock:
            self.hits += 1
            self.last_hit_time = time.monotonic()

    def seconds_since_last_hit(self) -> float:
        with self.lock:
            if self.last_hit_time == 0:
                return float("inf")
            return time.monotonic() - self.last_hit_time

    @property
    def total_hits(self):
        with self.lock:
            return self.hits


# ──────────── Single question processing ───────────────

def process_one(client, item: Dict, qid: str,
                args, rate_tracker: RateLimitTracker) -> Dict:
    question = item["question"]
    gt = item.get("reponse")
    if isinstance(gt, str):
        try:
            gt = float(gt)
        except ValueError:
            gt = None

    prompt = make_prompt(question)
    resp = None
    last_err = None
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        since_last = rate_tracker.seconds_since_last_hit()
        if since_last < 2.0:
            time.sleep(2.0 - since_last)
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
                log.error(f"FATAL: {last_err}")
                raise
            is_rl = "429" in last_err or "RESOURCE_EXHAUSTED" in last_err
            if is_rl:
                rate_tracker.record_hit()
                wait = min(5.0 * attempt, 30.0)
            else:
                wait = min(2.0 * attempt, 10.0)
            log.warning(f"err attempt {attempt}: {last_err[:120]} -> {wait}s")
            time.sleep(wait)

    if resp is None:
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": gt,
            "error": last_err,
            "model": args.model,
        }

    full_answer = (resp.text or "").strip()
    token_data = extract_token_logprobs(resp)
    if not token_data:
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": gt,
            "full_answer": full_answer,
            "error": "no logprobs returned",
            "model": args.model,
        }

    answer_raw, answer_value = parse_final_answer_text(full_answer)
    reasoning_td, answer_td = split_reasoning_answer(
        token_data, full_answer, answer_raw
    )

    return {
        "question_id": qid,
        "question": question,
        "ground_truth": gt,
        "full_answer": full_answer,
        "answer_text": answer_raw,
        "answer_value": answer_value,
        "model": args.model,
        "logprobs_k": args.logprobs_k,
        "temperature": args.temperature,
        "token_data": token_data,
        "reasoning_token_data": reasoning_td,
        "answer_token_data": answer_td,
        "n_tokens_total": len(token_data),
        "n_tokens_reasoning": len(reasoning_td),
        "n_tokens_answer": len(answer_td),
    }


# ──────────────────────────── Main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int,
                        default=DEFAULT_NUM_QUESTIONS)
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

    # Load GSM8K source
    src_items: List[Dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" not in obj or "reponse" not in obj:
                continue
            obj["_qid"] = f"gsm8k_{i:06d}"
            src_items.append(obj)
    src_items = src_items[:args.num_questions]
    print(f"Loaded {len(src_items)} GSM8K questions from {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_qids = set()
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "error" not in obj and "question_id" in obj:
                        done_qids.add(obj["question_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Already done: {len(done_qids)} (resuming).")

    todo = [it for it in src_items if it["_qid"] not in done_qids]
    print(f"To process: {len(todo)} (concurrency={args.concurrency})")

    write_lock = threading.Lock()
    rate_tracker = RateLimitTracker()
    semaphore = asyncio.Semaphore(args.concurrency)

    counters = {"done": 0, "ok": 0, "err": 0,
                "n_correct_so_far": 0, "n_parsed": 0,
                "tokens_total": 0, "t0": time.monotonic()}

    def report():
        elapsed = time.monotonic() - counters["t0"]
        rate = counters["done"] / elapsed if elapsed > 0 else 0.0
        eta = ((len(todo) - counters["done"]) / rate) if rate > 0 else float("inf")
        avg_tok = counters["tokens_total"] / max(1, counters["ok"])
        acc = (counters["n_correct_so_far"] / counters["n_parsed"]
               if counters["n_parsed"] else float("nan"))
        log.info(
            f"progress {counters['done']}/{len(todo)} "
            f"({100 * counters['done'] / max(1, len(todo)):.1f}%) "
            f"| ok={counters['ok']} err={counters['err']} "
            f"| parsed={counters['n_parsed']} acc(running)={acc:.3f} "
            f"| avg_tokens={avg_tok:.0f} "
            f"| {rate:.2f} q/s | ETA {eta / 60:.1f} min "
            f"| 429s={rate_tracker.total_hits}"
        )

    async def process_async(item, fout, pbar):
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, process_one, client, item, item["_qid"], args, rate_tracker
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
                    counters["tokens_total"] += result.get("n_tokens_total", 0)
                    pred = result.get("answer_value")
                    gt = result.get("ground_truth")
                    if pred is not None and gt is not None:
                        counters["n_parsed"] += 1
                        try:
                            if abs(float(pred) - float(gt)) <= 1e-3 + 1e-3 * abs(float(gt)):
                                counters["n_correct_so_far"] += 1
                        except (TypeError, ValueError):
                            pass
                if counters["done"] % 10 == 0 or counters["done"] == len(todo):
                    report()
            pbar.update(1)

    async def run_all():
        with open(out_path, "a", encoding="utf-8") as fout:
            pbar = tqdm(total=len(todo), desc="GSM8K top-K", unit="q",
                        file=sys.stderr, mininterval=2.0,
                        dynamic_ncols=True)
            tasks = [process_async(it, fout, pbar) for it in todo]
            chunk = args.concurrency * 4
            for i in range(0, len(tasks), chunk):
                await asyncio.gather(*tasks[i:i + chunk])
            pbar.close()

    t0 = time.monotonic()
    log.info(f"Starting GSM8K generation | {len(todo)} questions | "
             f"model={args.model} K={args.logprobs_k} "
             f"max_tokens={args.max_output_tokens} T={args.temperature} "
             f"concurrency={args.concurrency}")
    asyncio.run(run_all())
    log.info(f"Total: {time.monotonic() - t0:.0f}s "
             f"({rate_tracker.total_hits} rate limits)")

    # Summary
    n_total, n_err, n_parsed = 0, 0, 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            n_total += 1
            if "error" in obj:
                n_err += 1
            elif obj.get("answer_value") is not None:
                n_parsed += 1
    print(f"\nTotal entries: {n_total}  errors: {n_err}  "
          f"with parsed numeric answer: {n_parsed}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
