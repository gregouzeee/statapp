import json
import math
import os
import re
import time
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


# =========================
# Config (edit as needed)
# =========================
IN_JSONL = "triviaqa_numeric.jsonl"          # expects {"question": "...", "answer": 1945.0}
OUT_JSONL = "triviaqa_numeric_results.jsonl"

MODEL = "gemini-2.0-flash"
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 80

# IMPORTANT: logprobs_k = 1 (only the chosen token logprob at each step)
RESPONSE_LOGPROBS = True
LOGPROBS_K = 1
CANDIDATE_COUNT = 1

SLEEP = 0.0
RESUME = True
MAX_RETRIES = 5


# =========================
# Helpers: parsing & metrics
# =========================
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
EPS = 1e-9


def safe_float(s: str) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    if not NUM_RE.fullmatch(t):
        return None
    try:
        return float(t)
    except Exception:
        return None


def compute_errors(pred: float, true: float) -> Dict[str, float]:
    abs_error = abs(pred - true)
    rel_error = abs_error / max(abs(true), EPS)
    log_error = abs(math.log(abs(pred) + 1.0) - math.log(abs(true) + 1.0))
    return {"abs_error": abs_error, "rel_error": rel_error, "log_error": log_error}


def clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# =========================
# Prompt
# =========================
def make_numeric_prompt(question: str) -> str:
    q = question.strip()
    return f"""
You must answer in STRICT JSON with exactly two keys: "answer" and "confidence".

Rules:
- Return ONLY the JSON object. No extra text, no markdown, no explanation.
- "answer": a number in decimal notation (examples: 1945, 3.14, -12.5). No commas. No units. No text.
- "confidence": a float between 0 and 1 (inclusive). This number must be the confidence you have in the answer. If you respond 0.7 then
7 times out of 10 you are correct. Be careful, you must not overstate your confidence. It must be a JSON number, not a string.

Question:
{q}
""".strip()


# =========================
# Robust JSON extraction
# =========================
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_json(text: str) -> Dict[str, Any]:
    """
    Returns:
      {"answer_text": str, "answer_float": float, "confidence": float}
    Raises ValueError if invalid.
    """
    if text is None:
        raise ValueError("Empty response text")
    t = text.strip()

    # If any extra text slipped in, extract the first {...}
    if not (t.startswith("{") and t.endswith("}")):
        m = JSON_OBJ_RE.search(t)
        if m:
            t = m.group(0).strip()

    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise ValueError("Response is not a JSON object")

    if set(obj.keys()) != {"answer", "confidence"}:
        raise ValueError(f"Bad keys: {list(obj.keys())}")

    ans_raw = str(obj["answer"]).strip().replace(",", "")
    ans_val = safe_float(ans_raw)
    if ans_val is None:
        raise ValueError(f"Answer not numeric: {ans_raw}")

    conf = obj["confidence"]
    if not isinstance(conf, (int, float)):
        raise ValueError(f"Confidence not numeric: {conf}")
    conf = float(conf)
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"Confidence out of [0,1]: {conf}")

    return {"answer_text": ans_raw, "answer_float": ans_val, "confidence": conf}


# =========================
# Logprobs extraction (K=1, chosen token only)
# =========================
def iter_chosen_tokens_with_logprobs(resp) -> List[Tuple[str, float]]:
    """
    Returns list of (token_text, chosen_token_logp) in generation order.

    With logprobs=1, we only keep the single observed token's logprob at each step.
    Implementation still reads the per-step structure returned by the API.
    """
    cand = resp.candidates[0]
    lp = getattr(cand, "logprobs_result", None)
    if lp is None:
        raise RuntimeError("No logprobs_result returned (did you set response_logprobs=True?)")

    steps = getattr(lp, "top_candidates", None)
    if not steps:
        raise RuntimeError("No top_candidates in logprobs_result")

    out: List[Tuple[str, float]] = []
    for step in steps:
        cands = getattr(step, "candidates", None) or []
        if not cands:
            continue

        # With K=1, this is the chosen token only.
        chosen = cands[0]
        tok = getattr(chosen, "token", "")
        logp = getattr(chosen, "log_probability", None)
        if logp is None:
            continue
        out.append((str(tok), float(logp)))

    return out


def full_response_logprob_stats(resp) -> Dict[str, Any]:
    """
    Computes logprob stats for ALL generated tokens in the model output.
    Returns:
      {"logprob_sum": float|None, "logprob_avg": float|None, "n_tokens": int, "prob": float|None}
    prob = exp(logprob_sum), clamped to [0,1] with rule "if > 1.0 then 1.0".
    """
    toks = iter_chosen_tokens_with_logprobs(resp)
    if not toks:
        return {"logprob_sum": None, "logprob_avg": None, "n_tokens": 0, "prob": None}

    logps = [lp for _, lp in toks if lp is not None]
    if not logps:
        return {"logprob_sum": None, "logprob_avg": None, "n_tokens": 0, "prob": None}

    s = float(sum(logps))
    avg = s / len(logps)

    # Convert logprob sum -> probability of the exact token sequence
    p = float(math.exp(s)) if s > -1e300 else 0.0
    p = clamp01(p)

    return {"logprob_sum": s, "logprob_avg": avg, "n_tokens": len(logps), "prob": p}


def answer_logprob_stats(resp, answer_text: str) -> Dict[str, Any]:
    """
    Computes logprob stats for tokens that cover the answer substring in the model output.
    Best-effort: if we can't align answer in the output, returns None stats.
    Returns:
      {"logprob_sum": float|None, "logprob_avg": float|None, "n_tokens": int, "prob": float|None}
    prob = exp(logprob_sum), clamped to [0,1] with rule "if > 1.0 then 1.0".
    """
    full_text = (resp.text or "")
    if not full_text:
        return {"logprob_sum": None, "logprob_avg": None, "n_tokens": 0, "prob": None}

    # Find answer span (prefer the JSON "answer" field if possible)
    start = full_text.find(answer_text)
    if start < 0:
        m = re.search(r'"answer"\s*:\s*("?)(-?\d+(?:\.\d+)?)\1', full_text)
        if m:
            start = m.start(2)
            answer_text = m.group(2)
        else:
            return {"logprob_sum": None, "logprob_avg": None, "n_tokens": 0, "prob": None}
    end = start + len(answer_text)

    toks = iter_chosen_tokens_with_logprobs(resp)

    # Naive token->char span by concatenation
    spans: List[Tuple[int, int, str, float]] = []
    cursor = 0
    for tok, logp in toks:
        ts = cursor
        te = cursor + len(tok)
        spans.append((ts, te, tok, logp))
        cursor = te

    selected_logps: List[float] = []
    for ts, te, tok, logp in spans:
        if te <= start:
            continue
        if ts >= end:
            break
        selected_logps.append(logp)

    if not selected_logps:
        return {"logprob_sum": None, "logprob_avg": None, "n_tokens": 0, "prob": None}

    s = float(sum(selected_logps))
    avg = s / len(selected_logps)

    p = float(math.exp(s)) if s > -1e300 else 0.0
    p = clamp01(p)

    return {"logprob_sum": s, "logprob_avg": avg, "n_tokens": len(selected_logps), "prob": p}

def logps_to_metrics(logps: List[float]) -> Dict[str, Any]:
    """
    Convert a list of token log-probabilities into several interpretable metrics.
    """
    if not logps:
        return {
            "n_tokens": 0,
            "logprob_sum": None,
            "logprob_avg": None,
            "prob_joint": None,
            "prob_geo_mean": None,
            "perplexity": None,
            "p_min": None,
            "frac_p_gt_0_1": None,
            "frac_p_gt_0_01": None,
        }

    n = len(logps)
    s = float(sum(logps))
    avg = s / n

    # Joint probability of the exact sequence (can underflow to 0.0)
    prob_joint = float(math.exp(s)) if s > -1e300 else 0.0

    # Geometric mean token probability = exp(mean logp) (length-normalized)
    prob_geo_mean = float(math.exp(avg)) if avg > -1e300 else 0.0

    # Perplexity = exp(-mean logp)
    perplexity = float(math.exp(-avg)) if -avg < 1e300 else float("inf")

    # Per-token probabilities
    ps = [float(math.exp(lp)) if lp > -1e300 else 0.0 for lp in logps]
    p_min = min(ps) if ps else None

    frac_p_gt_0_1 = sum(1 for p in ps if p > 0.1) / n
    frac_p_gt_0_01 = sum(1 for p in ps if p > 0.01) / n

    # Optional clamp rule you mentioned (mostly matters for numerical weirdness)
    if prob_joint > 1.0:
        prob_joint = 1.0
    if prob_geo_mean > 1.0:
        prob_geo_mean = 1.0

    return {
        "n_tokens": n,
        "logprob_sum": s,
        "logprob_avg": avg,
        "prob_joint": prob_joint,
        "prob_geo_mean": prob_geo_mean,
        "perplexity": perplexity,
        "p_min": p_min,
        "frac_p_gt_0_1": frac_p_gt_0_1,
        "frac_p_gt_0_01": frac_p_gt_0_01,
    }


def all_tokens_metrics(resp) -> Dict[str, Any]:
    toks = iter_chosen_tokens_with_logprobs(resp)
    logps = [lp for _, lp in toks if lp is not None]
    return logps_to_metrics(logps)


def answer_tokens_metrics(resp, answer_text: str) -> Dict[str, Any]:
    full_text = (resp.text or "")
    if not full_text:
        return logps_to_metrics([])

    # Find answer span
    start = full_text.find(answer_text)
    if start < 0:
        m = re.search(r'"answer"\s*:\s*("?)(-?\d+(?:\.\d+)?)\1', full_text)
        if m:
            start = m.start(2)
            answer_text = m.group(2)
        else:
            return logps_to_metrics([])
    end = start + len(answer_text)

    toks = iter_chosen_tokens_with_logprobs(resp)

    # Token->char spans by concatenation (same as your current approach)
    spans: List[Tuple[int, int, str, float]] = []
    cursor = 0
    for tok, logp in toks:
        ts = cursor
        te = cursor + len(tok)
        spans.append((ts, te, tok, logp))
        cursor = te

    selected_logps: List[float] = []
    for ts, te, tok, logp in spans:
        if te <= start:
            continue
        if ts >= end:
            break
        selected_logps.append(logp)

    return logps_to_metrics(selected_logps)
# =========================
# Main
# =========================
def main():
    load_dotenv()
    PROJECT = os.getenv("GCP_PROJECT")
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    if PROJECT is None:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
    )

    # Resume logic
    done = set()
    if RESUME:
        try:
            with open(OUT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        key = obj.get("qid")
                        if key:
                            done.add(str(key))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

    # Read input
    with open(IN_JSONL, "r", encoding="utf-8") as fin:
        lines = [ln for ln in fin.read().splitlines() if ln.strip()]

    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for i, line in enumerate(tqdm(lines, desc="Run Gemini on numeric QA", unit="q")):
            item = json.loads(line)
            question = item.get("question")
            true_answer = item.get("answer")

            uid = item.get("uid")
            qid = str(uid) if uid is not None else f"{i:07d}"

            if RESUME and qid in done:
                continue

            if question is None or true_answer is None:
                out = {
                    "qid": qid,
                    "error": "missing_question_or_answer",
                    "item": item,
                    "model": MODEL,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            if not isinstance(true_answer, (int, float)):
                out = {
                    "qid": qid,
                    "error": "true_answer_not_numeric",
                    "question": question,
                    "answer_true_raw": true_answer,
                    "model": MODEL,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            prompt = make_numeric_prompt(question)

            # Call model with retries
            resp = None
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=TEMPERATURE,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            response_logprobs=RESPONSE_LOGPROBS,
                            logprobs=LOGPROBS_K,          # = 1
                            candidate_count=CANDIDATE_COUNT,
                        ),
                    )
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(min(2.0 * attempt, 10.0))

            if resp is None:
                out = {
                    "qid": qid,
                    "question": question,
                    "answer_true": float(true_answer),
                    "error": last_err,
                    "model": MODEL,
                    "logprobs_k": LOGPROBS_K,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            raw_text = resp.text or ""

            # Parse JSON
            try:
                parsed = parse_model_json(raw_text)
            except Exception as e:
                out = {
                    "qid": qid,
                    "question": question,
                    "answer_true": float(true_answer),
                    "model_answer_raw": raw_text,
                    "parse_error": str(e),
                    "model": MODEL,
                    "logprobs_k": LOGPROBS_K,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            pred_float = parsed["answer_float"]
            conf_verbalized = parsed["confidence"]

            # Proba sur tokens de la réponse (answer) uniquement
            metrics_answer = (
    answer_tokens_metrics(resp, parsed["answer_text"])
    if RESPONSE_LOGPROBS
    else logps_to_metrics([])
)

            metrics_all = (
    all_tokens_metrics(resp)
    if RESPONSE_LOGPROBS
    else logps_to_metrics([])
)  

            # Output demandé
            out = {
                "question": question,
                "answer_true": float(true_answer),
                "answer_model": pred_float,
                "confidence_verbalized": conf_verbalized,  # proba verbalisée par le LLM
                "prob_all_tokens": metrics_all,
                "prob_tokens_only": metrics_answer,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)


if __name__ == "__main__":
    main()