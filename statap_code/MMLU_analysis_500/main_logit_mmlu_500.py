import json
import math
import time
import unicodedata
from typing import Optional, Dict
import os
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


CHOICES = ["A", "B", "C", "D"]

HOMOGLYPH_MAP = {
    "А": "A",
    "В": "B",  
    "С": "C",  
    "Ｄ": "D",  
}


def token_to_choice(tok: str) -> Optional[str]:
    t = tok.strip()
    if not t:
        return None
    t = unicodedata.normalize("NFKC", t)  # Fullwidth -> ASCII when applicable
    t = HOMOGLYPH_MAP.get(t, t)
    return t if t in CHOICES else None


def make_qcm_prompt(item: dict) -> str:
    q = item["question"].strip()
    ch = item["choices"]
    return f"""
You must answer with EXACTLY ONE SINGLE LETTER among "A", "B", "C", "D".
Only ASCII uppercase letters are allowed (Unicode code points: U+0041, U+0042, U+0043, U+0044).
No other token is acceptable: no spaces, no newlines, no punctuation, no explanations.

Question:
{q}

A) {ch["A"]}
B) {ch["B"]}
C) {ch["C"]}
D) {ch["D"]}
""".strip()


def abcd_probs_including_all_candidates(resp) -> Dict[str, float]:
    """
    Step 0:
      P(letter) = sum_{token -> letter} exp(logp(token)) / sum_{token in topK} exp(logp(token))
    Returns probs for A/B/C/D only. Sum may be < 1 if other tokens exist in topK.
    """
    cand = resp.candidates[0]
    lp = getattr(cand, "logprobs_result", None)
    if lp is None or not lp.top_candidates:
        raise RuntimeError("No logprobs_result/top_candidates returned.")

    step0 = lp.top_candidates[0]
    cands = step0.candidates or []
    if not cands:
        raise RuntimeError("Empty top_candidates[0].")

    denom = sum(math.exp(c.log_probability) for c in cands)
    if denom <= 0.0:
        raise RuntimeError("Invalid denom from exp(logp).")

    num = {k: 0.0 for k in CHOICES}
    for c in cands:
        k = token_to_choice(c.token)
        if k is not None:
            num[k] += math.exp(c.log_probability)

    return {k: num[k] / denom for k in CHOICES}


def abcd_agg_logits_from_candidates(resp) -> Dict[str, Optional[float]]:
    """
    Aggregated 'logits' in your sense = log( sum exp(logp(token)) ) for tokens mapping to each letter.
    Returns None if no mapped token for that letter in topK.
    """
    cand = resp.candidates[0]
    lp = getattr(cand, "logprobs_result", None)
    if lp is None or not lp.top_candidates:
        raise RuntimeError("No logprobs_result/top_candidates returned.")

    step0 = lp.top_candidates[0]
    cands = step0.candidates or []
    if not cands:
        raise RuntimeError("Empty top_candidates[0].")

    sums = {k: 0.0 for k in CHOICES}
    for c in cands:
        k = token_to_choice(c.token)
        if k is not None:
            sums[k] += math.exp(c.log_probability)

    out: Dict[str, Optional[float]] = {}
    for k in CHOICES:
        out[k] = math.log(sums[k]) if sums[k] > 0.0 else None
    return out


def normalize_model_letter(text: str) -> Optional[str]:
    # Expected 1 token, but we normalize just in case
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    t = unicodedata.normalize("NFKC", t)
    t = HOMOGLYPH_MAP.get(t, t)
    if t in CHOICES:
        return t
    return None



def main():
    # --------- Load .env ----------
    load_dotenv()
    PROJECT = os.getenv("GCP_PROJECT")
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")

    if PROJECT is None:
        raise RuntimeError("GCP_PROJECT not found in .env")

    # --------- Config ----------
    IN_JSONL = "statap_code/logits_mmlu_500/mmlu_500.jsonl"
    OUT_JSONL = "statap_code/Conformal_prediction/logit_mmlu_500_temp0_5.jsonl"
    MODEL = "gemini-2.0-flash"
    LOGPROBS_K = 4
    SLEEP = 0.0
    RESUME = True
    MAX_RETRIES = 5

    # --------- Client ----------
    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
    )

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

        for line in tqdm(lines, desc="Run Gemini on MMLU JSONL", unit="q"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            uid = item.get("uid")

            if RESUME and uid in done:
                continue

            qcm_prompt = make_qcm_prompt(item)

            # ----- Call model with retry -----
            resp = None
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=qcm_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.5,
                            max_output_tokens=1,
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
                    "choices": item.get("choices"),
                    "solution": {
                        "answer_index": item.get("answer_index"),
                        "answer_letter": item.get("answer_letter"),
                        "answer_text": item.get("answer_text"),
                    },
                    "model": MODEL,
                    "logprobs_k": LOGPROBS_K,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            model_raw = resp.text or ""
            model_letter = normalize_model_letter(model_raw)

            try:
                probs = abcd_probs_including_all_candidates(resp)
                agg_logits = abcd_agg_logits_from_candidates(resp)
            except Exception as e:
                out = {
                    "uid": uid,
                    "question": item.get("question"),
                    "choices": item.get("choices"),
                    "solution": {
                        "answer_index": item.get("answer_index"),
                        "answer_letter": item.get("answer_letter"),
                        "answer_text": item.get("answer_text"),
                    },
                    "model_answer": {
                        "raw_text": model_raw,
                        "letter": model_letter,
                    },
                    "extract_error": str(e),
                    "model": MODEL,
                    "logprobs_k": LOGPROBS_K,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            out = {
                "uid": uid,
                "question": item["question"],
                "choices": item["choices"],
                "solution": {
                    "answer_index": item.get("answer_index"),
                    "answer_letter": item.get("answer_letter"),
                    "answer_text": item.get("answer_text"),
                    "subject": item.get("subject"),
                },
                "model_answer": {
                    "raw_text": model_raw,
                    "letter": model_letter,
                },
                "logits_abcd": agg_logits,
                "probs_abcd": probs,
                "model": MODEL,
                "logprobs_k": LOGPROBS_K,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)


if __name__ == "__main__":
    main()
