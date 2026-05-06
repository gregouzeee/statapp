"""
Semantic Entropy on GSM8K Chain-of-Thought generations
======================================================

Generates K=5 reasoning + answer pairs at T=1.0 per question, then
clusters them by:
    1) numeric equivalence on the parsed final `answer_value` (fast path)
    2) NLI fallback on the textual reasoning (slow path, rarely needed)

Computes Semantic Entropy = -Σ P(C) log P(C) over clusters, weighted
by length-normalised log-probability.

Output: data/se_gsm8k.jsonl with one line per question.
Joinable to gsm8k_judged.jsonl by question_id.
"""

import argparse
import asyncio
import json
import logging
import math
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    stream=sys.stdout)
log = logging.getLogger("se_gsm8k")


# ──────────────────────────── Config ────────────────────────────

GEN_MODEL = "gemini-2.0-flash"
NLI_MODEL = "gemini-2.5-flash"
K_SAMPLES = 5
TEMPERATURE = 1.0
LOGPROBS_K = 5
MAX_OUTPUT_TOKENS = 1024
NLI_MAX_TOKENS = 8192
NLI_BATCH_SIZE = 10
MAX_RETRIES = 4

ROOT = Path(__file__).resolve().parent
SOURCE_JSONL = ROOT.parent / "logit_gsm8k" / "dataset_gsm8k.jsonl"
OUT_JSONL = ROOT / "data" / "se_gsm8k.jsonl"


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


def make_prompt(q: str) -> str:
    return PROMPT_TEMPLATE.format(q=q.strip())


# ────────────────── Helpers ──────────────────

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_token_logprobs(resp) -> List[Dict]:
    tokens = []
    for cand in getattr(resp, "candidates", []) or []:
        lpr = getattr(cand, "logprobs_result", None)
        if not lpr:
            continue
        for cc in getattr(lpr, "chosen_candidates", []) or []:
            tokens.append({
                "token": cc.token,
                "log_prob": float(cc.log_probability),
            })
    return tokens


def parse_final_answer(text: str) -> Optional[float]:
    if not text:
        return None
    matches = list(re.finditer(r"\{[^{}]*\"answer\"[^{}]*\}", text))
    if matches:
        raw = matches[-1].group(0)
        try:
            obj = json.loads(raw)
            v = obj.get("answer")
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                m = _NUM_RE.search(v)
                if m:
                    return float(m.group(0))
        except Exception:
            pass
        m = _NUM_RE.search(raw)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = _NUM_RE.search(last)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return None


def _logsumexp(values: List[float]) -> float:
    if not values:
        return float("-inf")
    mx = max(values)
    if mx == float("-inf"):
        return float("-inf")
    return mx + math.log(sum(math.exp(v - mx) for v in values))


# ─────────── Generate one response with retries ────────────

def generate_one(client, question: str) -> Optional[Dict]:
    prompt = make_prompt(question)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_logprobs=True,
                    logprobs=LOGPROBS_K,
                    candidate_count=1,
                ),
            )
            text = (resp.text or "").strip()
            if not text:
                return None
            tokens = extract_token_logprobs(resp)
            if not tokens:
                return None
            log_prob_sum = sum(t["log_prob"] for t in tokens)
            return {
                "text": text,
                "answer_value": parse_final_answer(text),
                "log_prob_sum": log_prob_sum,
                "mean_log_prob": log_prob_sum / max(1, len(tokens)),
                "num_tokens": len(tokens),
            }
        except Exception as e:
            last_err = str(e)
            if any(f in last_err for f in ("Credentials", "BILLING",
                                           "PERMISSION_DENIED", "limit: 0")):
                raise
            time.sleep(min(2.0 * attempt, 10.0))
    return None


def generate_k(client, question: str, k: int) -> List[Dict]:
    out = []
    for _ in range(k):
        r = generate_one(client, question)
        if r is not None:
            out.append(r)
    return out


# ───────────── NLI clustering on reasoning text ─────────────

def _build_nli_prompt(pairs: List[Tuple[str, str, str]]) -> str:
    lines = [
        "You are a semantic equivalence judge for math reasoning.",
        "",
        "For each pair below, decide whether Reasoning A and Reasoning B "
        "lead to the same final numeric answer to the same problem.",
        "Two reasonings are equivalent if they reach the same number "
        "(allowing minor formatting differences). Different paths that "
        "still produce the same result count as equivalent.",
        "",
        "Return ONLY a JSON object in this exact format:",
        '{"equivalences": [true, false, true, ...]}',
        f"The array MUST have exactly {len(pairs)} elements, in the same order.",
        "No explanations, no markdown, just the JSON.",
        "",
    ]
    for idx, (q, a, b) in enumerate(pairs):
        lines.append(f"--- Pair {idx + 1} ---")
        lines.append(f"Problem: {q}")
        lines.append(f"Reasoning A: {a}")
        lines.append(f"Reasoning B: {b}")
        lines.append("")
    return "\n".join(lines)


def _parse_equivalences(raw: str, n: int) -> List[Optional[bool]]:
    if not raw:
        return [None] * n
    raw = re.sub(r"^```(?:json)?", "", raw.strip(),
                 flags=re.IGNORECASE | re.MULTILINE).strip()
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return [None] * n
    try:
        data = json.loads(m.group(0))
        arr = data.get("equivalences")
        if not isinstance(arr, list):
            return [None] * n
        out: List[Optional[bool]] = []
        for x in arr[:n]:
            out.append(x if isinstance(x, bool) else None)
        while len(out) < n:
            out.append(None)
        return out
    except Exception:
        return [None] * n


def cluster_responses(client, question: str,
                       responses: List[Dict]
                       ) -> Tuple[List[List[int]], List[Dict]]:
    n = len(responses)
    if n <= 1:
        return [[i] for i in range(n)], []

    parsed = [r.get("answer_value") for r in responses]
    if all(p is not None for p in parsed):
        groups: Dict[float, List[int]] = {}
        for i, v in enumerate(parsed):
            groups.setdefault(round(v, 6), []).append(i)
        return list(groups.values()), [{"method": "numeric_equality"}]

    # NLI fallback
    pairs_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tuples = [(question, responses[i]["text"], responses[j]["text"])
              for i, j in pairs_idx]
    eqs: List[Optional[bool]] = []
    for start in range(0, len(tuples), NLI_BATCH_SIZE):
        batch = tuples[start:start + NLI_BATCH_SIZE]
        prompt = _build_nli_prompt(batch)
        parsed_eq: Optional[List[Optional[bool]]] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=NLI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=NLI_MAX_TOKENS,
                        candidate_count=1,
                    ),
                )
                raw = (resp.text or "").strip()
                parsed_eq = _parse_equivalences(raw, len(batch))
                if parsed_eq and any(x is not None for x in parsed_eq):
                    break
            except Exception:
                time.sleep(min(2.0 * attempt, 10.0))
        if parsed_eq is None:
            parsed_eq = [None] * len(batch)
        eqs.extend(parsed_eq)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), eq in zip(pairs_idx, eqs):
        if eq is True:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    cmap: Dict[int, List[int]] = {}
    for i in range(n):
        cmap.setdefault(find(i), []).append(i)
    nli_pairs = [{"i": i, "j": j, "equivalent": eq}
                 for (i, j), eq in zip(pairs_idx, eqs)]
    return list(cmap.values()), nli_pairs


def compute_se(responses: List[Dict],
                clusters: List[List[int]]) -> float:
    if len(clusters) <= 1:
        return 0.0
    cluster_log_probs = [
        _logsumexp([responses[i]["mean_log_prob"] for i in c])
        for c in clusters
    ]
    log_z = _logsumexp(cluster_log_probs)
    se = 0.0
    for clp in cluster_log_probs:
        log_p = clp - log_z
        p = math.exp(log_p)
        if p > 0:
            se -= p * log_p
    return se


# ────────── Process one question ──────────

def process_one(client, item: Dict, qid: str) -> Dict:
    question = item["question"]
    gt = item.get("reponse")
    try:
        gt_f = float(gt) if gt is not None else None
    except (TypeError, ValueError):
        gt_f = None

    responses = generate_k(client, question, K_SAMPLES)
    if len(responses) < 2:
        return {
            "question_id": qid,
            "question": question,
            "ground_truth": gt_f,
            "error": f"only {len(responses)} responses",
            "n_responses": len(responses),
        }

    clusters, nli_pairs = cluster_responses(client, question, responses)
    se = compute_se(responses, clusters)

    cluster_log_probs = [
        _logsumexp([responses[i]["mean_log_prob"] for i in c])
        for c in clusters
    ]
    log_z = _logsumexp(cluster_log_probs) if cluster_log_probs else 0.0

    response_cluster = [0] * len(responses)
    cluster_details = []
    for cid, (c, clp) in enumerate(zip(clusters, cluster_log_probs)):
        prob = math.exp(clp - log_z) if log_z != float("-inf") else 0.0
        cluster_details.append({
            "cluster_id": cid,
            "response_indices": c,
            "probability": round(prob, 6),
        })
        for ri in c:
            response_cluster[ri] = cid

    return {
        "question_id": qid,
        "question": question,
        "ground_truth": gt_f,
        "semantic_entropy": round(se, 6),
        "num_clusters": len(clusters),
        "clusters": cluster_details,
        "responses": [
            {
                "answer_value": r.get("answer_value"),
                "log_prob_sum": round(r["log_prob_sum"], 4),
                "num_tokens": r["num_tokens"],
                "cluster_id": response_cluster[i],
                "text_preview": r["text"][:200],
            }
            for i, r in enumerate(responses)
        ],
        "nli_pairs": nli_pairs,
        "model": GEN_MODEL,
        "K_samples": K_SAMPLES,
        "temperature": TEMPERATURE,
    }


# ────────── Main ──────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=0,
                        help="Number of questions to process (0 = all)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input", type=str, default=str(SOURCE_JSONL))
    parser.add_argument("--output", type=str, default=str(OUT_JSONL))
    args = parser.parse_args()

    load_dotenv()
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")
    client = genai.Client(vertexai=True, project=project, location=location)

    src = []
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
            src.append(obj)
    if args.num > 0:
        src = src[:args.num]
    log.info(f"Loaded {len(src)} GSM8K questions from {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
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

    todo = [it for it in src if it["_qid"] not in done]
    log.info(f"To process: {len(todo)} (concurrency={args.concurrency})")

    write_lock = threading.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    counters = {"done": 0, "ok": 0, "err": 0,
                "se_sum": 0.0, "se_n": 0,
                "multi_cluster": 0, "t0": time.monotonic()}

    def report():
        elapsed = time.monotonic() - counters["t0"]
        rate = counters["done"] / elapsed if elapsed > 0 else 0.0
        eta = ((len(todo) - counters["done"]) / rate) if rate > 0 else float("inf")
        se_mean = (counters["se_sum"] / counters["se_n"]
                   if counters["se_n"] else float("nan"))
        log.info(
            f"progress {counters['done']}/{len(todo)} "
            f"({100 * counters['done'] / max(1, len(todo)):.1f}%) "
            f"| ok={counters['ok']} err={counters['err']} "
            f"| SE mean={se_mean:.3f} (>1 cluster: {counters['multi_cluster']}) "
            f"| {rate:.2f} q/s | ETA {eta / 60:.1f} min"
        )

    async def run_one(item, fout, pbar):
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, process_one, client, item, item["_qid"]
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
                    se = result.get("semantic_entropy")
                    if se is not None:
                        counters["se_sum"] += float(se)
                        counters["se_n"] += 1
                    if (result.get("num_clusters") or 0) > 1:
                        counters["multi_cluster"] += 1
                if counters["done"] % 25 == 0 or counters["done"] == len(todo):
                    report()
            pbar.update(1)

    async def run_all():
        with open(out_path, "a", encoding="utf-8") as fout:
            pbar = tqdm(total=len(todo), desc="SE GSM8K", unit="q",
                        file=sys.stderr, mininterval=2.0,
                        dynamic_ncols=True)
            tasks = [run_one(it, fout, pbar) for it in todo]
            chunk = args.concurrency * 4
            for i in range(0, len(tasks), chunk):
                await asyncio.gather(*tasks[i:i + chunk])
            pbar.close()

    log.info(f"Starting | model={GEN_MODEL} K={K_SAMPLES} T={TEMPERATURE} "
             f"concurrency={args.concurrency}")
    t0 = time.monotonic()
    asyncio.run(run_all())
    log.info(f"Total: {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
