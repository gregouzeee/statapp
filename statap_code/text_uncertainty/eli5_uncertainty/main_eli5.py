"""
ELI5 – Sentence-level uncertainty on long-form answers
=======================================================
1.  Load ELI5 questions (HuggingFace: rexarski/eli5_category).
2.  Prompt Gemini for "explain like I'm 5" answers with log-probs.
3.  Group tokens → words → sentences.
4.  Compute per-sentence uncertainty metrics:
        - Perplexity:     PPL(s) = exp(-mean(log_probs))
        - Top-k entropy:  H(s)   = mean of per-position Shannon entropies
5.  Save results to JSONL.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


# ──────────────────────────── Config ────────────────────────────

MODEL = "gemini-2.0-flash"
LOGPROBS_K = 5            # top-k candidates per position
MAX_OUTPUT_TOKENS = 512   # ELI5 answers are longer than TriviaQA
TEMPERATURE = 0.0
MAX_RETRIES = 5
SLEEP = 0.0
NUM_QUESTIONS = 100
RESUME = True

OUT_JSONL = Path(__file__).resolve().parent / "eli5_results.jsonl"


# ──────────────────────────── Prompt ────────────────────────────

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


# ──────────────── Log-prob extraction from Gemini ───────────────

def extract_token_logprobs(resp) -> List[Dict]:
    """
    Extract per-token info from Gemini's response:
      [{ "token": str, "log_prob": float,
         "top_k": [{"token": str, "log_prob": float}, ...] }, ...]
    """
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


# ──────────── Token → Word grouping ───────────────────────────

def group_tokens_into_words(token_data: List[Dict]) -> List[Dict]:
    """
    Merge sub-tokens into whole words.
    A token starting with a space begins a new word.
    """
    if not token_data:
        return []

    words: List[Dict] = []
    current_tokens: List[Dict] = []

    for t in token_data:
        tok_str = t["token"]
        if tok_str.startswith(" ") or not current_tokens:
            if current_tokens:
                words.append(_build_word(current_tokens))
            current_tokens = [t]
        else:
            current_tokens.append(t)

    if current_tokens:
        words.append(_build_word(current_tokens))

    return words


def _build_word(tokens: List[Dict]) -> Dict:
    word_str = "".join(t["token"] for t in tokens).strip()
    log_probs = [t["log_prob"] for t in tokens]
    return {
        "word": word_str,
        "tokens": tokens,
        "log_probs": log_probs,
        "sum_log_prob": sum(log_probs),
        "mean_log_prob": sum(log_probs) / len(log_probs),
        "min_log_prob": min(log_probs),
    }


# ──────────── Word → Sentence grouping ────────────────────────

SENTENCE_ENDERS = {'.', '!', '?'}


def group_words_into_sentences(words: List[Dict]) -> List[Dict]:
    """
    Split words into sentences.

    Boundary = word ends with '.', '!' or '?' AND either:
      - it's the last word, or
      - the next word starts with an uppercase letter.

    This avoids false splits on abbreviations like "U.S. government"
    (next word is lowercase).
    """
    if not words:
        return []

    sentences = []
    current_words: List[Dict] = []

    for i, w in enumerate(words):
        current_words.append(w)
        word_text = w["word"].rstrip()

        if word_text and word_text[-1] in SENTENCE_ENDERS:
            is_last = (i == len(words) - 1)
            if is_last:
                sentences.append(_build_sentence(current_words))
                current_words = []
            else:
                next_word = words[i + 1]["word"].lstrip()
                if next_word and next_word[0].isupper():
                    sentences.append(_build_sentence(current_words))
                    current_words = []

    # Flush remaining words as final sentence
    if current_words:
        sentences.append(_build_sentence(current_words))

    return sentences


def _build_sentence(words: List[Dict]) -> Dict:
    sentence_text = " ".join(w["word"] for w in words)
    all_tokens = []
    all_log_probs = []
    all_top_k = []
    for w in words:
        for t in w["tokens"]:
            all_tokens.append(t)
            all_log_probs.append(t["log_prob"])
            all_top_k.append(t.get("top_k", []))
    return {
        "sentence_text": sentence_text,
        "words": words,
        "tokens": all_tokens,
        "token_log_probs": all_log_probs,
        "token_top_k": all_top_k,
    }


# ──────────── Sentence-level metrics ──────────────────────────

def compute_sentence_perplexity(sentence: Dict) -> float:
    """PPL(s) = exp(-mean(log_probs of tokens in s))"""
    log_probs = sentence["token_log_probs"]
    if not log_probs:
        return 1.0
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)


def compute_sentence_topk_entropy(sentence: Dict) -> float:
    """
    Mean per-position Shannon entropy over top-k alternatives.

    For each token position:
      1. p_j = exp(log_prob_j)  for j in top-k
      2. Normalise: p_j = p_j / sum(p_j)
      3. H_i = -sum(p_j * ln(p_j))
    Sentence entropy = mean(H_i).
    """
    top_k_lists = sentence["token_top_k"]
    if not top_k_lists:
        return 0.0

    position_entropies = []
    for top_k in top_k_lists:
        if not top_k:
            position_entropies.append(0.0)
            continue

        log_probs = np.array([c["log_prob"] for c in top_k])
        probs = np.exp(log_probs)

        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(top_k)) / len(top_k)

        h = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
        position_entropies.append(float(h))

    return float(np.mean(position_entropies))


# ──────────── Dataset helpers ─────────────────────────────────

def get_reference_answer(item: Dict) -> str:
    """Return the reference answer from the ELI5 item."""
    return item.get("answer", "")


# ──────────────────────────── Main ──────────────────────────────

def main():
    load_dotenv()

    # ---- Client Gemini (Vertex AI) ----
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    # ---- Load ELI5 dataset ----
    from datasets import load_dataset

    print(f"Loading ELI5 dataset ({NUM_QUESTIONS} questions, streaming)...")
    ds_stream = load_dataset("sentence-transformers/eli5", split="train", streaming=True)
    items = list(ds_stream.take(NUM_QUESTIONS))
    print(f"Loaded {len(items)} questions.")

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
        for idx, item in enumerate(tqdm(items, desc="ELI5 + Gemini", unit="q")):
            qid = item.get("q_id", f"eli5_{idx}")

            if RESUME and qid in done:
                continue

            question = item["question"]
            reference_answer = get_reference_answer(item)

            print(f"\n{'=' * 70}")
            print(f"  Q {idx+1}/{len(items)}  [qid={qid}]")
            print(f"  Question: {question}")

            prompt = make_eli5_prompt(question)

            # ---- Call Gemini with retry ----
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
                    if "Credentials" in last_err or "credentials" in last_err:
                        print(f"\nAuth error: {last_err}")
                        print("Run: gcloud auth application-default login")
                        raise
                    if "BILLING" in last_err or "PERMISSION_DENIED" in last_err:
                        print(f"\nBilling/permission error: {last_err}")
                        raise
                    if "limit: 0" in last_err:
                        print(f"\nQuota exhausted (limit=0): {last_err}")
                        raise
                    print(f"  Retry {attempt}/{MAX_RETRIES}: {last_err}")
                    time.sleep(min(2.0 * attempt, 10.0))

            if resp is None:
                out = {
                    "question_id": qid, "question": question,
                    "error": last_err, "model": MODEL,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            full_answer = (resp.text or "").strip()
            print(f"  Answer: {full_answer[:120]}...")

            # ---- Extract token-level log-probs ----
            token_data = extract_token_logprobs(resp)
            if not token_data:
                out = {
                    "question_id": qid, "question": question,
                    "full_answer": full_answer,
                    "error": "no logprobs returned", "model": MODEL,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            # ---- Group tokens → words → sentences ----
            words = group_tokens_into_words(token_data)
            sentences = group_words_into_sentences(words)

            print(f"  {len(token_data)} tokens → {len(words)} words → {len(sentences)} sentences")

            # ---- Compute sentence-level metrics ----
            sentence_metrics = []
            for si, sent in enumerate(sentences):
                ppl = compute_sentence_perplexity(sent)
                topk_h = compute_sentence_topk_entropy(sent)
                avg_lp = (sum(sent["token_log_probs"]) / len(sent["token_log_probs"])
                          if sent["token_log_probs"] else 0.0)

                sentence_metrics.append({
                    "sentence_index": si,
                    "sentence_text": sent["sentence_text"],
                    "num_tokens": len(sent["tokens"]),
                    "num_words": len(sent["words"]),
                    "perplexity": ppl,
                    "topk_entropy": topk_h,
                    "avg_log_prob": avg_lp,
                })

                print(f"    S{si}: PPL={ppl:.2f}  H_topk={topk_h:.3f}  "
                      f"({len(sent['tokens'])} tok)  {sent['sentence_text'][:80]}...")

            # ---- Overall response metrics ----
            all_log_probs = [t["log_prob"] for t in token_data]
            overall_avg_logprob = sum(all_log_probs) / len(all_log_probs)
            overall_perplexity = math.exp(-overall_avg_logprob)
            overall_topk_entropy = compute_sentence_topk_entropy({
                "token_top_k": [t.get("top_k", []) for t in token_data],
                "token_log_probs": all_log_probs,
            })

            print(f"  Overall: PPL={overall_perplexity:.2f}  H_topk={overall_topk_entropy:.3f}")

            # ---- Build output record ----
            out = {
                "question_id": qid,
                "question": question,
                "reference_answer": reference_answer,
                "full_answer": full_answer,
                "model": MODEL,
                "logprobs_k": LOGPROBS_K,
                # Overall metrics
                "overall_perplexity": overall_perplexity,
                "overall_avg_log_prob": overall_avg_logprob,
                "overall_topk_entropy": overall_topk_entropy,
                # Sentence-level
                "num_sentences": len(sentences),
                "sentences": sentence_metrics,
                # Raw data (compact — no top_k to save space)
                "token_data": [
                    {"token": t["token"], "log_prob": t["log_prob"]}
                    for t in token_data
                ],
                "words": [
                    {"word": w["word"],
                     "sub_tokens": [t["token"] for t in w["tokens"]],
                     "sum_log_prob": w["sum_log_prob"]}
                    for w in words
                ],
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINISHED – Summary")
    print("=" * 70)
    print_summary(OUT_JSONL)


def print_summary(jsonl_path):
    """Print aggregate statistics from the results file."""
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

    # Overall perplexity
    perps = [r["overall_perplexity"] for r in results]
    print(f"Overall perplexity:  mean={np.mean(perps):.2f}  median={np.median(perps):.2f}")

    # Overall top-k entropy
    entropies = [r["overall_topk_entropy"] for r in results]
    print(f"Overall top-k entropy: mean={np.mean(entropies):.3f}  median={np.median(entropies):.3f}")

    # Sentence counts
    sent_counts = [r["num_sentences"] for r in results]
    print(f"Sentences per answer: mean={np.mean(sent_counts):.1f}  min={min(sent_counts)}  max={max(sent_counts)}")

    # Per-sentence perplexity
    all_sent_ppls = []
    for r in results:
        for s in r["sentences"]:
            all_sent_ppls.append(s["perplexity"])
    print(f"Per-sentence perplexity: mean={np.mean(all_sent_ppls):.2f}  median={np.median(all_sent_ppls):.2f}")

    # Per-sentence top-k entropy
    all_sent_h = []
    for r in results:
        for s in r["sentences"]:
            all_sent_h.append(s["topk_entropy"])
    print(f"Per-sentence top-k entropy: mean={np.mean(all_sent_h):.3f}  median={np.median(all_sent_h):.3f}")

    print()


if __name__ == "__main__":
    main()
