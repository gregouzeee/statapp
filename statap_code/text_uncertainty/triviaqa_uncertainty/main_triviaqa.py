"""
TriviaQA – Uncertainty on free-text answers
============================================
1.  Load TriviaQA (HuggingFace) and prompt Gemini for a one-sentence answer.
2.  Collect per-token log-probabilities from Gemini's response.
3.  Method A – Perplexity-based token importance:
        importance(t) = -log_prob(t)   (lower log-prob → more uncertain / more informative)
4.  Method B – Cosine-similarity-based token importance (using top-k log-prob vectors):
        importance(t) = 1 - cos(vec_full, vec_without_t)
5.  Compare the important tokens to the TriviaQA ground truth answer(s) using:
        - exact / fuzzy substring matching
        - token-level F1 (same metric as SQuAD / TriviaQA leaderboards)
"""

import json
import math
import os
import re
import time
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types


# ──────────────────────────── Config ────────────────────────────

MODEL = "gemini-2.0-flash"
LOGPROBS_K = 5            # top-k candidates per position
MAX_OUTPUT_TOKENS = 128
TEMPERATURE = 0.0
MAX_RETRIES = 5
SLEEP = 0.0
NUM_QUESTIONS = 100       # how many TriviaQA questions to process
RESUME = True
IMPORTANCE_TOP_K = 3      # how many "important" tokens to keep for comparison

IN_DATASET = "trivia_qa"  # HuggingFace dataset name
OUT_JSONL = "statap_code/text_uncertainty/triviaqa_uncertainty/triviaqa_results.jsonl"


# ──────────────────────────── Prompt ────────────────────────────

def make_triviaqa_prompt(question: str) -> str:
    return f"""Answer the following question in exactly ONE short sentence.
Your sentence must contain the answer.
Do not add any preamble, explanation, or follow-up.

Question: {question}
Answer:""".strip()


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
            print("    [extract] Candidate has no logprobs_result — skipping")
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

    # Print token table
    if tokens:
        print(f"    [extract] {len(tokens)} tokens extracted:")
        for i, t in enumerate(tokens):
            top_k_str = "  ".join(f"{c['token']!r}({c['log_prob']:.3f})" for c in t["top_k"])
            print(f"      pos {i:2d}: {t['token']!r:15s}  log_prob={t['log_prob']:.4f}  | top_k: {top_k_str}")
    else:
        print("    [extract] No tokens extracted!")
    return tokens


# ──────────── Token → Word grouping ───────────────────────────

def group_tokens_into_words(token_data: List[Dict]) -> List[Dict]:
    """
    Merge sub-tokens into whole words.

    Convention: a token that starts with a space (e.g. " the", " Bond") begins
    a new word. Tokens without a leading space (e.g. "das", "arian", "kei")
    are continuations of the previous word.

    Returns a list of word dicts:
      { "word": str,                   # reconstructed word (stripped)
        "tokens": [token_dict, ...],   # original token dicts that compose the word
        "log_probs": [float, ...],     # individual log-probs
        "sum_log_prob": float,         # sum of log-probs (= log P(word))
        "mean_log_prob": float,        # mean of log-probs
        "min_log_prob": float }        # min log-prob (= most uncertain sub-token)
    """
    if not token_data:
        return []

    words: List[Dict] = []
    current_tokens: List[Dict] = []

    for t in token_data:
        tok_str = t["token"]
        # A leading space or first token starts a new word
        if tok_str.startswith(" ") or not current_tokens:
            # flush previous word
            if current_tokens:
                words.append(_build_word(current_tokens))
            current_tokens = [t]
        else:
            current_tokens.append(t)

    # flush last word
    if current_tokens:
        words.append(_build_word(current_tokens))

    return words


def _build_word(tokens: List[Dict]) -> Dict:
    """Build a word dict from a list of consecutive sub-tokens."""
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


# ──────────── Word-level importance methods ───────────────────

def word_perplexity_importance(words: List[Dict]) -> List[Dict]:
    """
    Word importance = -sum_log_prob (sum of sub-token log-probs, negated).
    Filters out punctuation-only and whitespace-only words.
    """
    scored = []
    skipped = []
    for w in words:
        clean = w["word"].strip()
        if not clean or all(c in string.punctuation for c in clean):
            skipped.append(w["word"])
            continue
        scored.append({
            "word": w["word"],
            "sub_tokens": [t["token"] for t in w["tokens"]],
            "importance": -w["sum_log_prob"],
            "sum_log_prob": w["sum_log_prob"],
        })
    scored.sort(key=lambda x: x["importance"], reverse=True)

    print(f"    [word-perplexity] {len(scored)} words scored, {len(skipped)} skipped")
    print(f"    [word-perplexity] Ranking:")
    for rank, s in enumerate(scored, 1):
        marker = " <-- TOP" if rank <= IMPORTANCE_TOP_K else ""
        subtok_str = "+".join(f"{t!r}" for t in s["sub_tokens"])
        print(f"      #{rank:2d}  {s['word']!r:20s}  importance={s['importance']:.4f}  (tokens: {subtok_str}){marker}")
    return scored


def word_cosine_importance(words: List[Dict], token_data: List[Dict]) -> List[Dict]:
    """
    Cosine-similarity importance at word level.
    Same principle as token-level but removes all positions of a word at once.
    """
    if len(token_data) <= 1:
        print("    [word-cosine] <= 1 token — all importance = 0")
        return [{"word": w["word"], "importance": 0.0} for w in words]

    # Build shared vocabulary
    vocab = set()
    for t in token_data:
        for c in t.get("top_k", []):
            vocab.add(c["token"])
    vocab_index = {tok: i for i, tok in enumerate(sorted(vocab))}

    if not vocab_index:
        print("    [word-cosine] Empty vocabulary — all importance = 0")
        return [{"word": w["word"], "importance": 0.0} for w in words]

    # Per-position vectors
    position_vecs = [_topk_vector(t.get("top_k", []), vocab_index) for t in token_data]
    full_vec = np.mean(position_vecs, axis=0)

    # Map each token position to its word index
    token_to_word = []
    pos = 0
    for wi, w in enumerate(words):
        for _ in w["tokens"]:
            token_to_word.append(wi)
            pos += 1

    scored = []
    skipped = []
    for wi, w in enumerate(words):
        clean = w["word"].strip()
        if not clean or all(c in string.punctuation for c in clean):
            skipped.append(w["word"])
            continue
        # Remove ALL positions belonging to this word
        remaining = [v for j, v in enumerate(position_vecs) if token_to_word[j] != wi]
        if remaining:
            without_vec = np.mean(remaining, axis=0)
        else:
            without_vec = np.zeros_like(full_vec)
        sim = _cosine_sim(full_vec, without_vec)
        scored.append({
            "word": w["word"],
            "sub_tokens": [t["token"] for t in w["tokens"]],
            "importance": 1.0 - sim,
            "cosine_with_full": sim,
        })
    scored.sort(key=lambda x: x["importance"], reverse=True)

    print(f"    [word-cosine] {len(scored)} words scored, {len(skipped)} skipped")
    print(f"    [word-cosine] Ranking:")
    for rank, s in enumerate(scored, 1):
        marker = " <-- TOP" if rank <= IMPORTANCE_TOP_K else ""
        subtok_str = "+".join(f"{t!r}" for t in s["sub_tokens"])
        print(f"      #{rank:2d}  {s['word']!r:20s}  importance={s['importance']:.6f}  (tokens: {subtok_str}){marker}")
    return scored


def compare_important_words_to_answer(
    important_words: List[Dict],
    ground_truths: List[str],
    top_k: int = IMPORTANCE_TOP_K,
    method_name: str = "unknown",
) -> Dict:
    """Same as compare_important_tokens_to_answer but for words."""
    top_words = important_words[:top_k]

    print(f"    [compare-word {method_name}] Top-{top_k} words vs ground truths:")
    answer_rank = None
    for rank, w in enumerate(top_words, start=1):
        match = token_matches_any_truth(w["word"], ground_truths)
        status = "MATCH" if match else "no"
        print(f"      #{rank}  {w['word']!r:20s}  -> {status}")
        if match and answer_rank is None:
            answer_rank = rank

    if answer_rank is not None:
        print(f"    [compare-word {method_name}] => FOUND at rank {answer_rank}")
    else:
        print(f"    [compare-word {method_name}] => NOT FOUND in top-{top_k}")

    return {
        "answer_in_top_k": answer_rank is not None,
        "answer_rank": answer_rank,
        "top_words_text": [w["word"] for w in top_words],
    }


# ──────────────── Method A: Perplexity-based importance ─────────

def perplexity_importance(token_data: List[Dict]) -> List[Dict]:
    """
    Token importance = -log_prob (negated so higher = more important/uncertain).
    Returns list of {"token", "importance", "log_prob"} sorted by descending importance.
    """
    scored = []
    skipped = []
    for t in token_data:
        # skip pure whitespace / punctuation tokens
        if t["token"].strip() == "" or all(c in string.punctuation for c in t["token"].strip()):
            skipped.append(t["token"])
            continue
        scored.append({
            "token": t["token"],
            "importance": -t["log_prob"],
            "log_prob": t["log_prob"],
        })
    scored.sort(key=lambda x: x["importance"], reverse=True)

    print(f"    [perplexity] {len(scored)} scored, {len(skipped)} skipped (punct/ws)")
    print(f"    [perplexity] Ranking:")
    for rank, s in enumerate(scored, 1):
        marker = " <-- TOP" if rank <= IMPORTANCE_TOP_K else ""
        print(f"      #{rank:2d}  {s['token']!r:15s}  importance={s['importance']:.4f}  (log_prob={s['log_prob']:.4f}){marker}")
    return scored


# ──────────── Method B: Cosine-similarity importance ────────────

def _topk_vector(top_k: List[Dict], vocab_index: Dict[str, int]) -> np.ndarray:
    """Build a sparse-ish vector from top-k log-prob candidates at one position."""
    vec = np.full(len(vocab_index), -50.0)  # very low default log-prob
    for c in top_k:
        idx = vocab_index.get(c["token"])
        if idx is not None:
            vec[idx] = c["log_prob"]
    return vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def cosine_similarity_importance(token_data: List[Dict]) -> List[Dict]:
    """
    For each token position, build a log-prob vector over the shared top-k vocabulary.
    Full-sequence vector = mean of all position vectors.
    importance(t) = 1 - cos(full_vec, full_vec_without_t)

    Tokens whose removal changes the overall "log-prob profile" the most are important.
    """
    if len(token_data) <= 1:
        print("    [cosine] <= 1 token — all importance = 0")
        return [{"token": t["token"], "importance": 0.0} for t in token_data]

    # Build shared vocabulary from all top-k candidates across all positions
    vocab = set()
    for t in token_data:
        for c in t.get("top_k", []):
            vocab.add(c["token"])
    vocab_index = {tok: i for i, tok in enumerate(sorted(vocab))}
    print(f"    [cosine] Shared vocabulary size: {len(vocab_index)}")

    if not vocab_index:
        print("    [cosine] Empty vocabulary — all importance = 0")
        return [{"token": t["token"], "importance": 0.0} for t in token_data]

    # Build per-position vectors
    position_vecs = []
    for t in token_data:
        position_vecs.append(_topk_vector(t.get("top_k", []), vocab_index))

    full_vec = np.mean(position_vecs, axis=0)

    scored = []
    skipped = []
    for i, t in enumerate(token_data):
        # skip pure whitespace / punctuation
        if t["token"].strip() == "" or all(c in string.punctuation for c in t["token"].strip()):
            skipped.append(t["token"])
            continue
        # mean without position i
        remaining = [v for j, v in enumerate(position_vecs) if j != i]
        if remaining:
            without_vec = np.mean(remaining, axis=0)
        else:
            without_vec = np.zeros_like(full_vec)
        sim = _cosine_sim(full_vec, without_vec)
        scored.append({
            "token": t["token"],
            "importance": 1.0 - sim,
            "cosine_with_full": sim,
        })
    scored.sort(key=lambda x: x["importance"], reverse=True)

    print(f"    [cosine] {len(scored)} scored, {len(skipped)} skipped (punct/ws)")
    print(f"    [cosine] Ranking:")
    for rank, s in enumerate(scored, 1):
        marker = " <-- TOP" if rank <= IMPORTANCE_TOP_K else ""
        print(f"      #{rank:2d}  {s['token']!r:15s}  importance={s['importance']:.6f}  (cosine={s['cosine_with_full']:.6f}){marker}")
    return scored


# ──────────── Comparison with TriviaQA ground truth ─────────────

def normalize_answer(s: str) -> str:
    """Lower-case, remove articles, punctuation, and extra whitespace."""
    s = s.lower()
    # remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # collapse whitespace
    s = ' '.join(s.split())
    return s


def token_matches_any_truth(token: str, ground_truths: List[str]) -> bool:
    """Check if a single token (normalized) matches any ground truth answer."""
    norm_tok = normalize_answer(token)
    if not norm_tok:
        return False
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        # token is contained in the truth, or truth is contained in the token
        if norm_tok in norm_gt or norm_gt in norm_tok:
            return True
    return False


def compare_important_tokens_to_answer(
    important_tokens: List[Dict],
    ground_truths: List[str],
    top_k: int = IMPORTANCE_TOP_K,
    method_name: str = "unknown",
) -> Dict:
    """
    Simply check: is the answer among the top-k most important tokens?

    Returns:
      - answer_in_top_k: bool — is any ground truth found in the top-k tokens?
      - answer_rank: int|None — rank (1-based) of the first token matching the answer,
                                None if not found in top-k
      - top_tokens_text: list of the top-k token strings (for inspection)
    """
    top_tokens = important_tokens[:top_k]

    print(f"    [compare {method_name}] Top-{top_k} tokens vs ground truths:")
    answer_rank = None
    for rank, t in enumerate(top_tokens, start=1):
        match = token_matches_any_truth(t["token"], ground_truths)
        status = "MATCH" if match else "no"
        print(f"      #{rank}  {t['token'].strip()!r:15s}  -> {status}")
        if match and answer_rank is None:
            answer_rank = rank

    if answer_rank is not None:
        print(f"    [compare {method_name}] => FOUND at rank {answer_rank}")
    else:
        print(f"    [compare {method_name}] => NOT FOUND in top-{top_k}")

    return {
        "answer_in_top_k": answer_rank is not None,
        "answer_rank": answer_rank,
        "top_tokens_text": [t["token"].strip() for t in top_tokens],
    }


def evaluate_full_answer(full_answer: str, ground_truths: List[str]) -> bool:
    """Is any ground truth answer contained in the full LLM answer?"""
    norm_answer = normalize_answer(full_answer)
    print(f"    [eval] Normalized answer: {norm_answer!r}")
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if norm_gt in norm_answer:
            print(f"    [eval] => CORRECT (matched {gt!r})")
            return True
    print(f"    [eval] => INCORRECT (no ground truth found in answer)")
    return False


# ──────────────────────────── Main ──────────────────────────────

def main():
    load_dotenv()

    # ---- Client Gemini (Vertex AI) ----
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    # ---- Load TriviaQA (streaming to avoid downloading the full dataset) ----
    from datasets import load_dataset

    print(f"Loading TriviaQA dataset ({NUM_QUESTIONS} questions, streaming)...")
    ds_stream = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    ds = list(ds_stream.take(NUM_QUESTIONS))
    print(f"Loaded {len(ds)} questions.")

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
        for idx, item in enumerate(tqdm(ds, desc="TriviaQA + Gemini", unit="q")):
            qid = item["question_id"]

            if RESUME and qid in done:
                continue

            question = item["question"]
            # TriviaQA ground truths: aliases + normalized_aliases + value
            answer_obj = item["answer"]
            ground_truths = []
            if answer_obj.get("value"):
                ground_truths.append(answer_obj["value"])
            ground_truths.extend(answer_obj.get("aliases", []))
            ground_truths.extend(answer_obj.get("normalized_aliases", []))
            # deduplicate
            ground_truths = list(dict.fromkeys(ground_truths))

            print(f"\n{'=' * 70}")
            print(f"  Q {idx+1}/{len(ds)}  [qid={qid}]")
            print(f"  Question:      {question}")
            print(f"  Ground truths: {ground_truths}")

            prompt = make_triviaqa_prompt(question)
            print(f"  Prompt sent:\n    {prompt}")

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
                    # Fatal errors: fail immediately, no point retrying
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
                    print(f"\n  Retry {attempt}/{MAX_RETRIES}: {last_err}")
                    time.sleep(min(2.0 * attempt, 10.0))

            if resp is None:
                print(f"  ERROR: All {MAX_RETRIES} retries failed. Last error: {last_err}")
                out = {
                    "question_id": qid,
                    "question": question,
                    "ground_truths": ground_truths,
                    "error": last_err,
                    "model": MODEL,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            full_answer = (resp.text or "").strip()
            print(f"  LLM answer:    {full_answer!r}")

            # ---- Extract token-level log-probs ----
            print(f"\n  --- Token extraction ---")
            token_data = extract_token_logprobs(resp)

            if not token_data:
                print(f"  WARNING: No logprobs returned — skipping")
                out = {
                    "question_id": qid,
                    "question": question,
                    "ground_truths": ground_truths,
                    "full_answer": full_answer,
                    "error": "no logprobs returned",
                    "model": MODEL,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            # ---- Group tokens into words ----
            print(f"\n  --- Word grouping ---")
            words = group_tokens_into_words(token_data)
            print(f"    {len(token_data)} tokens -> {len(words)} words:")
            for wi, w in enumerate(words):
                subtoks = " + ".join(f"{t['token']!r}" for t in w["tokens"])
                print(f"      [{wi}] {w['word']!r:20s}  <- {subtoks}  (sum_lp={w['sum_log_prob']:.4f})")

            # ---- Token-level: Method A (Perplexity) ----
            print(f"\n  --- Token-level Method A: Perplexity importance ---")
            perp_scores = perplexity_importance(token_data)

            # ---- Token-level: Method B (Cosine) ----
            print(f"\n  --- Token-level Method B: Cosine similarity importance ---")
            cos_scores = cosine_similarity_importance(token_data)

            # ---- Word-level: Method A (Perplexity) ----
            print(f"\n  --- Word-level Method A: Perplexity importance ---")
            word_perp_scores = word_perplexity_importance(words)

            # ---- Word-level: Method B (Cosine) ----
            print(f"\n  --- Word-level Method B: Cosine similarity importance ---")
            word_cos_scores = word_cosine_importance(words, token_data)

            # ---- Is the full answer correct? ----
            print(f"\n  --- Full answer evaluation ---")
            answer_correct = evaluate_full_answer(full_answer, ground_truths)

            # ---- Token-level comparisons ----
            print(f"\n  --- Token-level top-{IMPORTANCE_TOP_K} comparison ---")
            perp_comparison = compare_important_tokens_to_answer(
                perp_scores, ground_truths, top_k=IMPORTANCE_TOP_K,
                method_name="perplexity",
            )
            cos_comparison = compare_important_tokens_to_answer(
                cos_scores, ground_truths, top_k=IMPORTANCE_TOP_K,
                method_name="cosine",
            )

            # ---- Word-level comparisons ----
            print(f"\n  --- Word-level top-{IMPORTANCE_TOP_K} comparison ---")
            word_perp_comparison = compare_important_words_to_answer(
                word_perp_scores, ground_truths, top_k=IMPORTANCE_TOP_K,
                method_name="perplexity",
            )
            word_cos_comparison = compare_important_words_to_answer(
                word_cos_scores, ground_truths, top_k=IMPORTANCE_TOP_K,
                method_name="cosine",
            )

            # ---- Sentence-level perplexity ----
            all_logprobs = [t["log_prob"] for t in token_data]
            avg_logprob = sum(all_logprobs) / len(all_logprobs)
            sentence_perplexity = math.exp(-avg_logprob)

            # ---- Summary for this example ----
            print(f"\n  --- SUMMARY ---")
            print(f"    Sentence perplexity: {sentence_perplexity:.4f}  (avg log_prob: {avg_logprob:.4f})")
            print(f"    Answer correct:      {answer_correct}")
            print(f"    TOKEN-LEVEL:")
            print(f"      Perplexity top-{IMPORTANCE_TOP_K}: {perp_comparison['top_tokens_text']}  -> match={perp_comparison['answer_in_top_k']} (rank={perp_comparison['answer_rank']})")
            print(f"      Cosine top-{IMPORTANCE_TOP_K}:     {cos_comparison['top_tokens_text']}  -> match={cos_comparison['answer_in_top_k']} (rank={cos_comparison['answer_rank']})")
            print(f"    WORD-LEVEL:")
            print(f"      Perplexity top-{IMPORTANCE_TOP_K}: {word_perp_comparison['top_words_text']}  -> match={word_perp_comparison['answer_in_top_k']} (rank={word_perp_comparison['answer_rank']})")
            print(f"      Cosine top-{IMPORTANCE_TOP_K}:     {word_cos_comparison['top_words_text']}  -> match={word_cos_comparison['answer_in_top_k']} (rank={word_cos_comparison['answer_rank']})")

            out = {
                "question_id": qid,
                "question": question,
                "ground_truths": ground_truths,
                "full_answer": full_answer,
                "answer_correct": answer_correct,
                "sentence_perplexity": sentence_perplexity,
                "avg_log_prob": avg_logprob,
                "token_data": [
                    {"token": t["token"], "log_prob": t["log_prob"]}
                    for t in token_data
                ],
                "words": [
                    {"word": w["word"], "sub_tokens": [t["token"] for t in w["tokens"]],
                     "sum_log_prob": w["sum_log_prob"]}
                    for w in words
                ],
                "perplexity_importance": {
                    "top_tokens": perp_scores[:IMPORTANCE_TOP_K],
                    "comparison": perp_comparison,
                },
                "cosine_importance": {
                    "top_tokens": cos_scores[:IMPORTANCE_TOP_K],
                    "comparison": cos_comparison,
                },
                "word_perplexity_importance": {
                    "top_words": word_perp_scores[:IMPORTANCE_TOP_K],
                    "comparison": word_perp_comparison,
                },
                "word_cosine_importance": {
                    "top_words": word_cos_scores[:IMPORTANCE_TOP_K],
                    "comparison": word_cos_comparison,
                },
                "model": MODEL,
                "logprobs_k": LOGPROBS_K,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

            if SLEEP > 0:
                time.sleep(SLEEP)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINISHED – Generating summary statistics")
    print("=" * 70)
    print_summary(OUT_JSONL)


def print_summary(jsonl_path: str):
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

    # Answer accuracy
    correct_count = sum(1 for r in results if r["answer_correct"])
    print(f"\nTotal valid results: {n}")
    print(f"Answer correct (truth in answer): {correct_count}/{n} ({correct_count/n:.1%})")

    # Perplexity stats
    avg_perp = np.mean([r["sentence_perplexity"] for r in results])
    print(f"Avg sentence perplexity:          {avg_perp:.2f}")

    # Helper to print hit-rate for a given key path
    def _print_hit_rate(label, results, key, sub_key="comparison"):
        hits = sum(1 for r in results if key in r and r[key][sub_key]["answer_in_top_k"])
        ranks = [r[key][sub_key]["answer_rank"]
                 for r in results
                 if key in r and r[key][sub_key]["answer_rank"] is not None]
        total = sum(1 for r in results if key in r)
        if total == 0:
            return
        print(f"\n{label} (top-{IMPORTANCE_TOP_K}):")
        print(f"  Answer found:    {hits}/{total} ({hits/total:.1%})")
        if ranks:
            print(f"  Avg rank when found: {np.mean(ranks):.1f}")

    _print_hit_rate("Token-level perplexity", results, "perplexity_importance")
    _print_hit_rate("Token-level cosine", results, "cosine_importance")
    _print_hit_rate("Word-level perplexity", results, "word_perplexity_importance")
    _print_hit_rate("Word-level cosine", results, "word_cosine_importance")

    # Correlation: perplexity vs correctness
    perplexities = [r["sentence_perplexity"] for r in results]
    correctness = [1.0 if r["answer_correct"] else 0.0 for r in results]
    if len(set(correctness)) > 1:
        from scipy.stats import pointbiserialr
        corr, pval = pointbiserialr(correctness, perplexities)
        print(f"\nCorrelation (perplexity vs correctness): r={corr:.3f}, p={pval:.4f}")

    print()


if __name__ == "__main__":
    main()
