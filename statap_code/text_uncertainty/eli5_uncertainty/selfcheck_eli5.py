"""
ELI5 – SelfCheck (adapted from SelfCheckGPT, Manakul et al., 2023)
====================================================================
For each of the 100 ELI5 questions:
1.  Take the original answer's sentences from eli5_judged.jsonl.
2.  Take the K=5 alternative responses from eli5_semantic_entropy.jsonl.
3.  For each (sentence, passage) pair, ask Gemini 2.5 Flash:
    "Is this sentence supported by the passage?" → Yes/No.
4.  SelfCheck score per sentence = proportion of "No" answers.
5.  Overall score = mean of sentence scores (0=reliable, 1=contradicted).
6.  Save to eli5_selfcheck.jsonl.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


# ──────────────────────────── Config ────────────────────────────

CHECKER_MODEL = "gemini-2.5-flash"
CHECKER_MAX_TOKENS = 8192
MAX_RETRIES = 5
NUM_QUESTIONS = 100
RESUME = True

JUDGED_JSONL = Path(__file__).resolve().parent / "eli5_judged.jsonl"
SE_JSONL = Path(__file__).resolve().parent / "eli5_semantic_entropy.jsonl"
OUT_JSONL = Path(__file__).resolve().parent / "eli5_selfcheck.jsonl"


# ──────────── Build SelfCheck prompt ──────────────────────────

def _build_selfcheck_prompt(sentences: List[str], passage: str) -> str:
    """
    Build a batched selfcheck prompt: given a passage, evaluate N sentences.
    """
    lines = [
        "You are a factual consistency checker.",
        "",
        "Given a reference passage, determine if each sentence is supported by it.",
        "A sentence is 'supported' if the reference passage conveys the same "
        "information or is consistent with the sentence.",
        "A sentence is 'not supported' if the reference passage contradicts it "
        "or does not mention the relevant information at all.",
        "",
        f"Reference passage: {passage}",
        "",
        "For each sentence below, answer whether it is supported.",
        f"Return ONLY a JSON object: {{\"supported\": [true, false, ...]}}",
        f"The array MUST have exactly {len(sentences)} boolean elements.",
        "No explanations, no markdown fences, just the JSON.",
        "",
    ]

    for idx, sentence in enumerate(sentences):
        lines.append(f"Sentence {idx + 1}: {sentence}")

    return "\n".join(lines)


# ──────────── Parse SelfCheck response ────────────────────────

def _parse_supported(raw_text: str, expected_len: int) -> List[Optional[bool]]:
    """Parse the checker's JSON response {"supported": [true, false, ...]}."""
    if not raw_text:
        return [None] * expected_len

    # Strip markdown fences
    cleaned = re.sub(
        r"^(```json|```|''')\\s*", "", raw_text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(
        r"(```|''')\\s*$", "", cleaned.strip(), flags=re.MULTILINE
    )

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        print(f"    [SelfCheck] WARNING: No JSON found: {raw_text[:200]}")
        return [None] * expected_len

    try:
        data = json.loads(m.group(0))
        arr = data.get("supported")
        if not isinstance(arr, list):
            print(f"    [SelfCheck] WARNING: No 'supported' list: {data}")
            return [None] * expected_len

        out: List[Optional[bool]] = []
        for x in arr[:expected_len]:
            if isinstance(x, bool):
                out.append(x)
            else:
                out.append(None)

        while len(out) < expected_len:
            out.append(None)

        return out

    except Exception as e:
        print(f"    [SelfCheck] WARNING: JSON parse error: {e}")
        return [None] * expected_len


# ──────────── SelfCheck for one question ──────────────────────

def selfcheck_question(
    client,
    sentences: List[str],
    passages: List[str],
) -> Dict:
    """
    Run SelfCheck for one question.

    For each passage, check all sentences in one batched call.
    Build matrix scores[n_sentences][k_passages]:
        True (supported) → 0.0
        False (not supported) → 1.0
        None (parse error) → 0.5

    Returns dict with sentence_scores and overall_score.
    """
    n_sent = len(sentences)
    k_pass = len(passages)

    # scores matrix: rows=sentences, cols=passages
    scores = [[0.5] * k_pass for _ in range(n_sent)]

    for pass_idx, passage in enumerate(passages):
        prompt = _build_selfcheck_prompt(sentences, passage)
        parsed = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=CHECKER_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=CHECKER_MAX_TOKENS,
                        candidate_count=1,
                    ),
                )
                raw = (resp.text or "").strip()
                parsed = _parse_supported(raw, expected_len=n_sent)
                if parsed is not None and any(x is not None for x in parsed):
                    break
                print(f"    [SelfCheck] Parse failed attempt {attempt}, retrying...")
            except Exception as e:
                err_str = str(e)
                if "Credentials" in err_str or "credentials" in err_str:
                    raise
                if "BILLING" in err_str or "PERMISSION_DENIED" in err_str:
                    raise
                print(f"    [SelfCheck] API error attempt {attempt}: {e}")
                time.sleep(min(2.0 * attempt, 10.0))

        if parsed is None:
            parsed = [None] * n_sent

        # Fill scores matrix for this passage
        for sent_idx, supported in enumerate(parsed):
            if supported is True:
                scores[sent_idx][pass_idx] = 0.0  # supported → low score
            elif supported is False:
                scores[sent_idx][pass_idx] = 1.0  # not supported → high score
            else:
                scores[sent_idx][pass_idx] = 0.5  # unknown

    # Compute per-sentence scores (average over passages)
    sentence_scores = []
    for sent_idx in range(n_sent):
        avg = sum(scores[sent_idx]) / k_pass
        sentence_scores.append(round(avg, 4))

    # Overall score = mean of sentence scores
    overall_score = sum(sentence_scores) / n_sent if n_sent > 0 else 0.0

    return {
        "sentence_scores": sentence_scores,
        "overall_score": round(overall_score, 4),
    }


# ──────────────────────────── Main ──────────────────────────────

def main():
    load_dotenv()

    # ---- Client Gemini (Vertex AI) ----
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    # ---- Load data ----
    print(f"Loading judged data from {JUDGED_JSONL}...")
    judged = {}
    with open(JUDGED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" not in obj and "question_id" in obj:
                judged[obj["question_id"]] = obj

    print(f"Loaded {len(judged)} judged questions.")

    print(f"Loading SE data from {SE_JSONL}...")
    se_data = {}
    with open(SE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" not in obj and "question_id" in obj:
                se_data[obj["question_id"]] = obj

    print(f"Loaded {len(se_data)} SE questions.")

    # ---- Merge: find common question_ids ----
    common_qids = sorted(set(judged.keys()) & set(se_data.keys()))
    print(f"Common questions: {len(common_qids)}")

    if not common_qids:
        print("No common questions found. Exiting.")
        return

    # ---- Resume logic ----
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
                        if "question_id" in obj:
                            done.add(obj["question_id"])
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

    print(f"Already processed: {len(done)} questions. Resuming...")

    # ---- Main loop ----
    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for idx, qid in enumerate(tqdm(common_qids, desc="SelfCheck", unit="q")):
            if RESUME and qid in done:
                continue

            j = judged[qid]
            s = se_data[qid]
            question = j["question"]

            # Extract sentences from judged data
            sentence_texts = [
                sent["sentence_text"]
                for sent in j.get("sentences", [])
                if sent.get("sentence_text")
            ]

            if not sentence_texts:
                print(f"  Q {idx + 1}: No sentences found, skipping.")
                out = {
                    "question_id": qid,
                    "question": question,
                    "error": "no sentences",
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            # Extract K alternative passages from SE data
            passages = [
                r["text"]
                for r in s.get("responses", [])
                if r.get("text")
            ]

            if not passages:
                print(f"  Q {idx + 1}: No passages found, skipping.")
                out = {
                    "question_id": qid,
                    "question": question,
                    "error": "no passages",
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            print(f"\n{'=' * 70}")
            print(f"  Q {idx + 1}/{len(common_qids)}  [qid={qid}]")
            print(f"  Question: {question[:80]}")
            print(f"  Sentences: {len(sentence_texts)}, Passages: {len(passages)}")

            # Run SelfCheck
            result = selfcheck_question(client, sentence_texts, passages)

            print(f"  SelfCheck score: {result['overall_score']:.4f}")
            for i, (sent, sc) in enumerate(
                zip(sentence_texts, result["sentence_scores"])
            ):
                label = "OK" if sc < 0.3 else ("WARN" if sc < 0.7 else "BAD")
                print(f"    S{i} [{label}] score={sc:.2f}: {sent[:60]}...")

            # Save
            out = {
                "question_id": qid,
                "question": question,
                "selfcheck_score": result["overall_score"],
                "num_sentences": len(sentence_texts),
                "sentence_scores": result["sentence_scores"],
                "sentence_texts": sentence_texts,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINISHED – Summary")
    print("=" * 70)
    print_summary(OUT_JSONL)


def print_summary(jsonl_path: Path):
    """Print aggregate statistics from the SelfCheck results file."""
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" not in obj:
                results.append(obj)

    if not results:
        print("No valid results to summarize.")
        return

    n = len(results)
    print(f"\nTotal valid results: {n}")

    scores = [r["selfcheck_score"] for r in results]
    import numpy as np
    print(f"SelfCheck score:  mean={np.mean(scores):.4f}  "
          f"median={np.median(scores):.4f}  "
          f"std={np.std(scores):.4f}")
    print(f"                  min={min(scores):.4f}  "
          f"max={max(scores):.4f}")

    # Distribution of scores in buckets
    low = sum(1 for s in scores if s < 0.3)
    mid = sum(1 for s in scores if 0.3 <= s < 0.7)
    high = sum(1 for s in scores if s >= 0.7)
    print(f"  Reliable (< 0.3):     {low:3d}")
    print(f"  Uncertain (0.3-0.7):  {mid:3d}")
    print(f"  Unreliable (>= 0.7):  {high:3d}")
    print()


if __name__ == "__main__":
    main()
