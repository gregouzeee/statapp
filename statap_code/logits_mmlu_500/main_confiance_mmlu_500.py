import json
import math
import re
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

def normalize_choice_key(k: str) -> Optional[str]:
    if k is None:
        return None
    t = str(k).strip()
    if not t:
        return None
    t = unicodedata.normalize("NFKC", t)
    t = HOMOGLYPH_MAP.get(t, t)
    return t if t in CHOICES else None

def strip_code_fences(raw_text: str) -> str:
    # Enlève ```json / ``` / ''' au début et à la fin (comme ton exemple)
    cleaned = re.sub(
        r"^(\s*```json|\s*```|\s*''')\s*",
        "",
        (raw_text or "").strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(
        r"(\s*```|\s*''')\s*$",
        "",
        cleaned.strip(),
        flags=re.MULTILINE,
    )
    return cleaned.strip()

def parse_confidence_json(raw_text: str) -> Dict[str, float]:
    cleaned = strip_code_fences(raw_text)
    obj = json.loads(cleaned)  # parse direct
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object.")
    # Normalize keys + cast floats
    out = {k: 0.0 for k in CHOICES}
    for k, v in obj.items():
        kk = normalize_choice_key(k)
        if kk is None:
            continue
        fv = float(v)  # accepte "0.2" aussi
        if not math.isfinite(fv):
            fv = 0.0
        out[kk] += fv
    return out

def normalize_to_probs(conf: Dict[str, float]) -> Dict[str, float]:
    # clip à >=0 puis renormalise
    clipped = {k: max(0.0, float(conf.get(k, 0.0) or 0.0)) for k in CHOICES}
    s = sum(clipped.values())
    if s <= 0.0:
        return {k: 1.0 / len(CHOICES) for k in CHOICES}
    return {k: clipped[k] / s for k in CHOICES}

def argmax_letter_from_probs(probs: Dict[str, float]) -> str:
    return max(CHOICES, key=lambda k: probs.get(k, 0.0))

def make_qcm_prompt(item: dict) -> str:
    q = item["question"].strip()
    ch = item["choices"]
    return f"""
Return ONLY a JSON object with keys "A","B","C","D" and float values (confidence for each option). No extra text.
Question:
{q}

A) {ch["A"]}
B) {ch["B"]}
C) {ch["C"]}
D) {ch["D"]}
""".strip()

def main():
    load_dotenv()
    PROJECT = os.getenv("GCP_PROJECT")
    LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    if PROJECT is None:
        raise RuntimeError("GCP_PROJECT not found in .env")

    IN_JSONL = "statap_code/logits_mmlu_500/mmlu_500.jsonl"
    OUT_JSONL = "statap_code/Conformal_prediction/mmlu_500_confidence_probs.jsonl"
    MODEL = "gemini-2.0-flash"
    TEMPERATURE = 0.5
    SLEEP = 0.0
    RESUME = True
    MAX_RETRIES = 5

    client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

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

    with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "a", encoding="utf-8") as fout:
        lines = fin.readlines()

        for line in tqdm(lines, desc="Run Gemini (confidence JSON)", unit="q"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            uid = item.get("uid")
            if RESUME and uid in done:
                continue

            prompt = make_qcm_prompt(item)

            resp = None
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=TEMPERATURE,
                            response_mime_type="application/json",  # aide beaucoup, mais on strip quand même
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
                        "subject": item.get("subject"),
                    },
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            raw_text = resp.text or ""

            try:
                conf = parse_confidence_json(raw_text)     # dict A/B/C/D (pas normalisé)
                probs = normalize_to_probs(conf)           # dict A/B/C/D somme=1
                model_letter = argmax_letter_from_probs(probs)
            except Exception as e:
                out = {
                    "uid": uid,
                    "extract_error": str(e),
                    "model_raw": raw_text,
                    "question": item.get("question"),
                    "choices": item.get("choices"),
                    "solution": {
                        "answer_index": item.get("answer_index"),
                        "answer_letter": item.get("answer_letter"),
                        "answer_text": item.get("answer_text"),
                        "subject": item.get("subject"),
                    },
                    "model": MODEL,
                    "temperature": TEMPERATURE,
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
                    "raw_text": raw_text,
                    "letter": model_letter,
                },
                "probs_abcd": probs,
                "model": MODEL,
                "temperature": TEMPERATURE,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)

if __name__ == "__main__":
    main()
