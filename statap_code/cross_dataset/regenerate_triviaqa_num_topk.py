"""
Regenerate TriviaQA-numeric answers with top-K log-probabilities
================================================================

The original `triviaqa_numeric_results.jsonl` (used in §3) only stores
aggregated log-prob statistics — not the per-token top-K distribution.
EPR and WEPR therefore cannot be computed on it.

This script re-runs the same prompt as §3 on the same questions but
with `logprobs_k=10`, so we can score each generation with EPR /
WEPR / SE (via cross_table.py).

Output: data/triviaqa_num_topk.jsonl with one line per question:

    {
      "question": str,
      "answer_true": float,
      "answer_text": str,
      "answer_value": float|None,
      "confidence_verbalized": float|None,
      "model": str,
      "logprobs_k": int,
      "temperature": float,
      "token_data": [
        {"token": str, "log_prob": float,
         "top_k": [{"token": str, "log_prob": float}, ...]}, ...
      ]
    }

The `question` and `answer_true` fields can be used to LEFT-join with
the existing aggregates of `triviaqa_numeric_results.jsonl`.
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
log = logging.getLogger("regen_triviaqa_num")
log.setLevel(logging.INFO)


# ──────────────────────────── Config ────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_LOGPROBS_K = 10
DEFAULT_MAX_OUTPUT_TOKENS = 64
DEFAULT_TEMPERATURE = 0.0   # match §3 default behaviour: deterministic
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_RETRIES = 5

ROOT = Path(__file__).resolve().parent
SOURCE_JSONL = (ROOT.parent / "distance_verite"
                / "triviaqa_numeric_results.jsonl")
OUT_JSONL = ROOT / "data" / "triviaqa_num_topk.jsonl"


PROMPT_TEMPLATE = """
You must answer in STRICT JSON with exactly two keys: "answer" and "confidence".

Rules:
- Return ONLY the JSON object. No extra text, no markdown, no explanation.
- "answer": a number in decimal notation (examples: 1945, 3.14, -12.5). No commas. No units. No text.
- "confidence": a float between 0 and 1 (inclusive). This number must be the confidence you have in the answer. If you respond 0.7 then
7 times out of 10 you are correct. Be careful, you must not overstate your confidence. It must be a JSON number, not a string.

Question:
{q}
""".strip()


def make_prompt(q: str) -> str:
    return PROMPT_TEMPLATE.format(q=q.strip())


# ──────────────────── log-prob extraction ────────────────────

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


# ──────────────────── JSON answer parsing ────────────────────

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_json(text: str) -> Tuple[Optional[str],
                                          Optional[float],
                                          Optional[float]]:
    """
    Returns (answer_text, answer_value, confidence).
    """
    if not text:
        return None, None, None
    m = _JSON_RE.search(text)
    if m is None:
        # Fallback: any number
        nm = _NUM_RE.search(text)
        if nm:
            try:
                return text.strip(), float(nm.group(0)), None
            except ValueError:
                pass
        return None, None, None
    raw = m.group(0)
    try:
        obj = json.loads(raw)
        a = obj.get("answer")
        c = obj.get("confidence")
        ans_text = str(a) if a is not None else None
        ans_val: Optional[float] = None
        if isinstance(a, (int, float)):
            ans_val = float(a)
        elif isinstance(a, str):
            nm = _NUM_RE.search(a)
            if nm:
                try:
                    ans_val = float(nm.group(0))
                except ValueError:
                    pass
        conf: Optional[float] = None
        if isinstance(c, (int, float)):
            conf = float(c)
        elif isinstance(c, str):
            try:
                conf = float(c)
            except ValueError:
                pass
        return ans_text, ans_val, conf
    except Exception:
        nm = _NUM_RE.search(raw)
        if nm:
            try:
                return raw, float(nm.group(0)), None
            except ValueError:
                pass
    return None, None, None


# ──────────────────── Rate limit tracker ────────────────────

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


# ──────────────────── Process one question ────────────────────

def process_one(client, item: Dict, args,
                rate_tracker: RateLimitTracker) -> Dict:
    question = item["question"]
    answer_true = item.get("answer_true")
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
            "question": question,
            "answer_true": answer_true,
            "error": last_err,
            "model": args.model,
        }

    full_text = (resp.text or "").strip()
    token_data = extract_token_logprobs(resp)
    if not token_data:
        return {
            "question": question,
            "answer_true": answer_true,
            "error": "no logprobs",
            "model": args.model,
        }

    ans_text, ans_val, conf = parse_model_json(full_text)
    return {
        "question": question,
        "answer_true": answer_true,
        "answer_text": ans_text,
        "answer_value": ans_val,
        "confidence_verbalized": conf,
        "model": args.model,
        "logprobs_k": args.logprobs_k,
        "temperature": args.temperature,
        "token_data": token_data,
        "n_tokens": len(token_data),
    }


# ──────────────────────────── Main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=0,
                        help="Number of questions (0 = all available)")
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

    # Load source items (just questions + truths)
    src: List[Dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" not in obj:
                continue
            src.append({
                "question": obj["question"],
                "answer_true": obj.get("answer_true"),
            })
    if args.num > 0:
        src = src[:args.num]
    log.info(f"Loaded {len(src)} questions from {args.input}")

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
                    if "error" not in obj and "question" in obj:
                        done.add(obj["question"])
                except json.JSONDecodeError:
                    continue
    log.info(f"Already done: {len(done)} (resuming).")

    todo = [it for it in src if it["question"] not in done]
    log.info(f"To process: {len(todo)} (concurrency={args.concurrency})")

    write_lock = threading.Lock()
    rate_tracker = RateLimitTracker()
    semaphore = asyncio.Semaphore(args.concurrency)

    counters = {"done": 0, "ok": 0, "err": 0,
                "n_correct": 0, "n_parsed": 0,
                "tokens_total": 0, "t0": time.monotonic()}

    def report():
        elapsed = time.monotonic() - counters["t0"]
        rate = counters["done"] / elapsed if elapsed > 0 else 0.0
        eta = ((len(todo) - counters["done"]) / rate) if rate > 0 else float("inf")
        avg_tok = counters["tokens_total"] / max(1, counters["ok"])
        acc = (counters["n_correct"] / counters["n_parsed"]
               if counters["n_parsed"] else float("nan"))
        log.info(
            f"progress {counters['done']}/{len(todo)} "
            f"({100 * counters['done'] / max(1, len(todo)):.1f}%) "
            f"| ok={counters['ok']} err={counters['err']} "
            f"| parsed={counters['n_parsed']} acc={acc:.3f} "
            f"| avg_tok={avg_tok:.0f} "
            f"| {rate:.2f} q/s | ETA {eta / 60:.1f} min "
            f"| 429s={rate_tracker.total_hits}"
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
                    pred = result.get("answer_value")
                    gt = result.get("answer_true")
                    if pred is not None and gt is not None:
                        counters["n_parsed"] += 1
                        try:
                            if abs(float(pred) - float(gt)) <= 1e-3 + 1e-3 * abs(float(gt)):
                                counters["n_correct"] += 1
                        except (TypeError, ValueError):
                            pass
                if counters["done"] % 50 == 0 or counters["done"] == len(todo):
                    report()
            pbar.update(1)

    async def run_all():
        with open(out_path, "a", encoding="utf-8") as fout:
            pbar = tqdm(total=len(todo), desc="TriviaQA-num topK", unit="q",
                        file=sys.stderr, mininterval=2.0,
                        dynamic_ncols=True)
            tasks = [process_async(it, fout, pbar) for it in todo]
            chunk = args.concurrency * 4
            for i in range(0, len(tasks), chunk):
                await asyncio.gather(*tasks[i:i + chunk])
            pbar.close()

    log.info(f"Starting | model={args.model} K={args.logprobs_k} "
             f"T={args.temperature} concurrency={args.concurrency}")
    t0 = time.monotonic()
    asyncio.run(run_all())
    log.info(f"Total: {time.monotonic() - t0:.0f}s "
             f"({rate_tracker.total_hits} rate limits)")


if __name__ == "__main__":
    main()
