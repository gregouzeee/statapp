import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


# =========================
# Config (edit as needed)
# =========================
IN_JSONL = "triviaqa_numeric_results.jsonl"          # output of the previous script
OUT_JSONL = "output.jsonl"

JUDGE_MODEL = "gemini-2.0-flash"
JUDGE_TEMPERATURE = 0.5
JUDGE_MAX_OUTPUT_TOKENS = 60

SLEEP = 0.0
RESUME = True
MAX_RETRIES = 5


# =========================
# Prompt + parsing
# =========================
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)




def make_judge_prompt(question: str, model_answer_text: str, verbal_conf: Optional[float] = None) -> str:
    """
    Judge sees only the question + the model's answer.
    Judge must output a calibrated probability in [0,1] of correctness.
    """
    return f"""
You are an assistant that evaluates the correctness of a model's answer to a question. 
Return a single JSON object with exactly one key: "judge_confidence".
- "judge_confidence" must be a JSON number in [0, 1].
- Return ONLY the JSON object. No markdown, no extra text.

The judge confidence must be the probaility that the model's answer is correct, calibrated as best as possible.
If the model's answer is very likely correct, return a value close to 1. If it's likely incorrect, return a value close to 0. If you are uncertain, return something in between.
You must not return 0 or 1 unless you are very certain. Avoid returning extreme values. Be more partial.
Question:
{question.strip()}

Model answer (numeric text):
{model_answer_text.strip()}
""".strip()


def parse_judge_json(text: str) -> float:
    if text is None:
        raise ValueError("Empty judge response text")
    t = text.strip()

    if not (t.startswith("{") and t.endswith("}")):
        m = JSON_OBJ_RE.search(t)
        if m:
            t = m.group(0).strip()

    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise ValueError("Judge response is not a JSON object")

    if set(obj.keys()) != {"judge_confidence"}:
        raise ValueError(f"Bad keys in judge JSON: {list(obj.keys())}")

    val = obj["judge_confidence"]
    if not isinstance(val, (int, float)):
        raise ValueError("judge_confidence is not numeric")
    val = float(val)
    if not (0.0 <= val <= 1.0):
        raise ValueError("judge_confidence out of [0,1]")
    return val


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

    # Resume logic: skip items already written to OUT_JSONL
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
                        # With the new schema, there may be no qid; fallback to question hash-ish key
                        key = obj.get("qid") or obj.get("question")
                        if key:
                            done.add(str(key))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

    with open(IN_JSONL, "r", encoding="utf-8") as fin:
        lines = [ln for ln in fin.read().splitlines() if ln.strip()]

    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="LLM-as-a-judge", unit="q"):
            item = json.loads(line)

            # New schema key
            question = item.get("question")
            key = str(item.get("qid") or question or "")

            if RESUME and key and key in done:
                continue

            # If previous step had errors / missing fields, carry forward
            if not question:
                item["judge"] = {"error": "missing_question"}
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            # New schema: answer_model is a float, but judge wants numeric text
            ans_model = item.get("answer_model")
            if ans_model is None or not isinstance(ans_model, (int, float)):
                item["judge"] = {"error": "missing_or_non_numeric_answer_model"}
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            ans_text = str(float(ans_model)).rstrip("0").rstrip(".") if isinstance(ans_model, float) else str(ans_model)

            # New schema: confidence_verbalized
            verbal_conf = item.get("confidence_verbalized", None)
            if not isinstance(verbal_conf, (int, float)):
                verbal_conf = None

            prompt = make_judge_prompt(question, ans_text, verbal_conf)

            # Call judge with retries
            resp = None
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = client.models.generate_content(
                        model=JUDGE_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=JUDGE_TEMPERATURE,
                            max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                            candidate_count=1,
                        ),
                    )
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(min(2.0 * attempt, 10.0))

            if resp is None:
                item["judge"] = {
                    "error": last_err,
                    "model": JUDGE_MODEL,
                    "temperature": JUDGE_TEMPERATURE,
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            raw_text = resp.text or ""
            import math

            EPS = 1e-9

            raw_text = resp.text or ""

            # --- Parse judge score ---
            try:
                judge_score = parse_judge_json(raw_text)
            except Exception as e:
                judge_score = None
                judge_error = str(e)

            # --- Compute distance ---
            true_val = item.get("answer_true")
            pred_val = item.get("answer_model")

            if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
                abs_error = abs(float(pred_val) - float(true_val))
                rel_error = abs_error / max(abs(float(true_val)), EPS)
                log_error = abs(
                    math.log(abs(float(pred_val)) + 1.0)
                    - math.log(abs(float(true_val)) + 1.0)
                )
            else:
                abs_error = None
                rel_error = None
                log_error = None

            out = dict(item) 

            out["judge_confidence"] = judge_score
            out["abs_error"] = abs_error
            out["rel_error"] = rel_error
            out["log_error"] = log_error

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)


if __name__ == "__main__":
    main()