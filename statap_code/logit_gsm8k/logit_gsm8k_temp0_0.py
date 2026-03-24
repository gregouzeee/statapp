import json
import math
import time
import unicodedata
import re
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

# -----------------------------
# Strict float parsing
# -----------------------------
FLOAT_RE = re.compile(r"^-?\d+\.\d+$")  # requires decimal point


def parse_strict_float(text: str) -> Optional[float]:
    if text is None:
        return None
    t = unicodedata.normalize("NFKC", text.strip())
    if not t:
        return None
    if not FLOAT_RE.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def make_float_prompt(item: dict) -> str:
    q = item["question"].strip()
    return  f"""
You must answer with EXACTLY ONE Float number. Only the number must be in the output.

Question:
{q}
""".strip()


# -----------------------------
# Softmax-topK confidence (always)
# -----------------------------
import math

def dump_all_topk_logprobs(resp, drop_whitespace_steps=True):
    """
    Returns a heavy dict containing, for each decoding step:
      - chosen token + raw score
      - topK candidates: token, token_id, raw_score, prob_hat (softmax over topK)
    Also returns aggregated confidence (p_geo_hat etc.) computed from chosen tokens.

    drop_whitespace_steps=True will SKIP steps where the chosen token is whitespace ('\\n', ' ', '\\t').
    """
    cand = resp.candidates[0]
    lp = getattr(cand, "logprobs_result", None)
    if lp is None or not getattr(lp, "top_candidates", None):
        raise RuntimeError("No logprobs_result/top_candidates. Need response_logprobs=True.")

    steps = lp.top_candidates

    per_step = []
    sum_logp_hat = 0.0
    used_steps = 0
    tokens_used = []

    for step_idx, step in enumerate(steps):
        cands = getattr(step, "candidates", None) or []
        if not cands:
            continue

        # chosen candidate if available, else fallback to first candidate
        chosen = getattr(step, "chosen_candidate", None)
        if chosen is None:
            chosen = cands[0]

        chosen_tok = getattr(chosen, "token", "")
        chosen_score = getattr(chosen, "log_probability", None)

        # Optionally skip whitespace steps (like newline)
        if drop_whitespace_steps and chosen_tok.strip() == "":
            continue

        # Collect raw scores for softmax
        raw_scores = []
        for c in cands:
            s = getattr(c, "log_probability", None)
            if s is None:
                continue
            raw_scores.append(float(s))

        if chosen_score is None or not raw_scores:
            # Still dump what we can
            per_step.append({
                "step_idx": step_idx,
                "chosen_token": chosen_tok,
                "chosen_token_id": getattr(chosen, "token_id", None),
                "chosen_raw_score": float(chosen_score) if chosen_score is not None else None,
                "softmax_over_topk": None,
                "topk_candidates": [
                    {
                        "token": getattr(c, "token", None),
                        "token_id": getattr(c, "token_id", None),
                        "raw_score": float(getattr(c, "log_probability")) if getattr(c, "log_probability", None) is not None else None,
                        "prob_hat": None,
                    }
                    for c in cands
                ],
            })
            continue

        chosen_score = float(chosen_score)

        # Stable softmax over topK
        m = max(raw_scores)
        denom = sum(math.exp(s - m) for s in raw_scores)

        topk_list = []
        chosen_prob_hat = None

        for c in cands:
            tok = getattr(c, "token", None)
            tok_id = getattr(c, "token_id", None)
            s = getattr(c, "log_probability", None)
            if s is None:
                topk_list.append({"token": tok, "token_id": tok_id, "raw_score": None, "prob_hat": None})
                continue

            s = float(s)
            p_hat = math.exp(s - m) / denom
            p_hat = max(min(p_hat, 1.0), 1e-300)

            topk_list.append({
                "token": tok,
                "token_id": tok_id,
                "raw_score": s,
                "prob_hat": p_hat,
            })

            if tok == chosen_tok and tok_id == getattr(chosen, "token_id", tok_id):
                chosen_prob_hat = p_hat

        # If matching by (token, token_id) failed, fallback: compute from chosen_score directly
        if chosen_prob_hat is None:
            chosen_prob_hat = math.exp(chosen_score - m) / denom
            chosen_prob_hat = max(min(chosen_prob_hat, 1.0), 1e-300)

        # Aggregate confidence
        sum_logp_hat += math.log(chosen_prob_hat)
        used_steps += 1
        tokens_used.append(chosen_tok)

        per_step.append({
            "step_idx": step_idx,
            "chosen_token": chosen_tok,
            "chosen_token_id": getattr(chosen, "token_id", None),
            "chosen_raw_score": chosen_score,
            "chosen_prob_hat": chosen_prob_hat,
            "softmax_over_topk": {
                "max_raw_score": m,
                "denom_exp_shifted": denom
            },
            "topk_candidates": topk_list,
        })

    if used_steps == 0:
        raise RuntimeError("No usable steps after filtering; try drop_whitespace_steps=False")

    avg_logp_hat = sum_logp_hat / used_steps

    return {
        "tokens_used": tokens_used,
        "n_tokens": used_steps,
        "sum_logprob_hat": sum_logp_hat,
        "avg_logprob_hat": avg_logp_hat,
        "p_seq_hat": math.exp(sum_logp_hat),
        "p_geo_hat": math.exp(avg_logp_hat),
        "per_step": per_step,  # HEAVY: contains ALL candidates with raw_score + prob_hat
    }


def main():
    # --------- Load .env ----------
    load_dotenv()
    PROJECT = os.getenv("GCP_PROJECT")
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    if PROJECT is None:
        raise RuntimeError("GCP_PROJECT not found in .env")

    # --------- Config ----------
    IN_JSONL = "statap_code/logit_gsm8k/test.jsonl"
    OUT_JSONL = "statap_code/logit_gsm8k/out_float_softmax_conf.jsonl"

    MODEL = "gemini-2.0-flash"
    LOGPROBS_K = 5              # topK candidates per step (softmax normalizes over these)
    MAX_OUTPUT_TOKENS = 16
    TEMPERATURE = 0.0           # recommended for clean numeric outputs

    SLEEP = 0.0
    RESUME = True
    MAX_RETRIES = 5

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

    # --------- Resume logic ----------
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
                        if "uid" in obj:
                            done.add(obj["uid"])
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

    # --------- Main loop ----------
    with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "a", encoding="utf-8") as fout:
        lines = fin.readlines()

        for line in tqdm(lines, desc="Run Gemini (float + softmax conf)", unit="q"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            uid = item.get("uid")

            if RESUME and uid in done:
                continue

            prompt = make_float_prompt(item)

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
                            response_logprobs=True,
                            logprobs=LOGPROBS_K,
                            candidate_count=1,
                        ),
                    )
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(min(2.0 * attempt, 10.0))

            if resp is None:
                out = {
                    "uid": uid,
                    "error": last_err,
                    "question": item.get("question"),
                    "model": MODEL,
                    "logprobs_k": LOGPROBS_K,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            model_raw = unicodedata.normalize("NFKC", (resp.text or "").strip())
            value = parse_strict_float(model_raw)

            try:
                conf = dump_all_topk_logprobs(resp, drop_whitespace_steps=True)
            except Exception as e:
                out = {
                    "uid": uid,
                    "question": item.get("question"),
                    "model_answer": {"raw_text": model_raw, "value": value},
                    "confidence_error": str(e),
                    "model": MODEL,
                    "logprobs_k": LOGPROBS_K,
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            out = {
    "uid": uid,
    "question": item.get("question"),
    "model_answer": {"raw_text": model_raw, "value": value},
    "confidence": conf,   # <-- IMPORTANT: write the full heavy object including per_step
    "model": MODEL,
    "logprobs_k": LOGPROBS_K,
    "temperature": TEMPERATURE,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
}


            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)


if __name__ == "__main__":
    main()
