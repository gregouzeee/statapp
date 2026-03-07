"""
ELI5 – Semantic Entropy (Kuhn et al., 2023)
=============================================
1.  Load the same 100 ELI5 questions used in main_eli5.py.
2.  Generate K=5 responses per question with temperature=0.7 (+ log-probs).
3.  Cluster responses by semantic equivalence using NLI (Gemini 2.5 Flash).
4.  Compute Semantic Entropy = Shannon entropy over cluster probabilities.
5.  Save results to eli5_semantic_entropy.jsonl.
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

# Reuse prompt builder and logprob extraction from main_eli5
from main_eli5 import make_eli5_prompt, extract_token_logprobs


# ──────────────────────────── Config ────────────────────────────

MODEL = "gemini-2.0-flash"           # generation (needs logprobs)
NLI_MODEL = "gemini-2.5-flash"       # semantic clustering
K_SAMPLES = 5                        # responses per question
TEMPERATURE = 1.0                    # high temp for diverse responses
LOGPROBS_K = 5
MAX_OUTPUT_TOKENS = 512
MAX_RETRIES = 5
NLI_MAX_TOKENS = 8192               # 2.5 Flash needs headroom for thinking
NLI_BATCH_SIZE = 10                  # NLI comparisons per API call
NUM_QUESTIONS = 100
RESUME = True

OUT_JSONL = Path(__file__).resolve().parent / "eli5_semantic_entropy.jsonl"


# ──────────── Generate K responses ──────────────────────────────

def generate_k_responses(
    client,
    question: str,
    k: int = K_SAMPLES,
) -> List[Dict]:
    """
    Generate K responses for a question with temperature > 0.
    Returns list of {"text": str, "log_prob_sum": float, "num_tokens": int}.
    """
    prompt = make_eli5_prompt(question)
    responses = []

    for sample_idx in range(k):
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
                    raise
                if "BILLING" in last_err or "PERMISSION_DENIED" in last_err:
                    raise
                if "limit: 0" in last_err:
                    raise
                print(f"    Retry {attempt}/{MAX_RETRIES} (sample {sample_idx}): "
                      f"{last_err}")
                time.sleep(min(2.0 * attempt, 10.0))

        if resp is None:
            print(f"    Sample {sample_idx}: FAILED after {MAX_RETRIES} retries")
            continue

        text = (resp.text or "").strip()
        if not text:
            print(f"    Sample {sample_idx}: empty response")
            continue

        # Extract token log-probs
        token_data = extract_token_logprobs(resp)
        if not token_data:
            print(f"    Sample {sample_idx}: no logprobs")
            continue

        log_prob_sum = sum(t["log_prob"] for t in token_data)
        mean_log_prob = log_prob_sum / len(token_data)

        responses.append({
            "text": text,
            "log_prob_sum": log_prob_sum,
            "mean_log_prob": mean_log_prob,
            "num_tokens": len(token_data),
        })

    return responses


# ──────────── NLI-based semantic clustering ─────────────────────

def _build_nli_prompt(pairs: List[Tuple[str, str, str]]) -> str:
    """
    Build a batched NLI prompt.
    Each pair = (question, response_A, response_B).
    Ask: do A and B convey the same meaning?
    """
    lines = [
        "You are a semantic equivalence judge.",
        "",
        "For each pair of answers below, determine if they convey the SAME "
        "meaning in response to the given question.",
        "Two answers are equivalent if a reader would learn the same core "
        "information from both, even if wording differs.",
        "Minor differences in phrasing, analogies, or additional details are OK "
        "— focus on whether the key explanation is the same.",
        "",
        "Return ONLY a JSON object in this exact format:",
        '{"equivalences": [true, false, true, ...]}',
        f"The array MUST have exactly {len(pairs)} elements, in the same order.",
        "No explanations, no markdown fences, just the JSON.",
        "",
    ]

    for idx, (question, resp_a, resp_b) in enumerate(pairs):
        lines.append(f"--- Pair {idx + 1} ---")
        lines.append(f"Question: {question}")
        lines.append(f"Answer A: {resp_a}")
        lines.append(f"Answer B: {resp_b}")
        lines.append("")

    return "\n".join(lines)


def _parse_equivalences(
    raw_text: str, expected_len: int
) -> List[Optional[bool]]:
    """Parse the NLI judge's JSON response."""
    if not raw_text:
        return [None] * expected_len

    # Strip markdown fences
    cleaned = re.sub(
        r"^(```json|```|''')\s*", "", raw_text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(
        r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE
    )

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not m:
        print(f"    [NLI] WARNING: No JSON found: {raw_text[:200]}")
        return [None] * expected_len

    try:
        data = json.loads(m.group(0))
        arr = data.get("equivalences")
        if not isinstance(arr, list):
            print(f"    [NLI] WARNING: No 'equivalences' list: {data}")
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
        print(f"    [NLI] WARNING: JSON parse error: {e}")
        return [None] * expected_len


def cluster_responses_nli(
    client,
    question: str,
    responses: List[Dict],
) -> Tuple[List[List[int]], List[Dict]]:
    """
    Cluster responses by semantic equivalence using NLI.

    Returns:
        clusters: list of clusters, each a list of response indices
        nli_pairs: list of {"i": int, "j": int, "equivalent": bool|None}
    """
    n = len(responses)
    if n <= 1:
        return [[i] for i in range(n)], []

    # Generate all pairs (i, j) with i < j
    all_pairs_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs_indices.append((i, j))

    # Build NLI query tuples
    nli_tuples = [
        (question, responses[i]["text"], responses[j]["text"])
        for i, j in all_pairs_indices
    ]

    # Call NLI in batches
    equivalences: List[Optional[bool]] = []

    for start in range(0, len(nli_tuples), NLI_BATCH_SIZE):
        end = min(start + NLI_BATCH_SIZE, len(nli_tuples))
        batch = nli_tuples[start:end]

        prompt = _build_nli_prompt(batch)
        parsed = None

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
                parsed = _parse_equivalences(raw, expected_len=len(batch))
                if parsed is not None and any(x is not None for x in parsed):
                    break
                print(f"    [NLI] Parse failed attempt {attempt}, retrying...")
            except Exception as e:
                print(f"    [NLI] API error attempt {attempt}: {e}")
                time.sleep(min(2.0 * attempt, 10.0))

        if parsed is None:
            parsed = [None] * len(batch)

        equivalences.extend(parsed)

    # Record NLI pair results
    nli_pairs = []
    for (i, j), eq in zip(all_pairs_indices, equivalences):
        nli_pairs.append({"i": i, "j": j, "equivalent": eq})

    # Union-Find clustering
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (i, j), eq in zip(all_pairs_indices, equivalences):
        if eq is True:
            union(i, j)

    # Build clusters from union-find
    cluster_map: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        cluster_map.setdefault(root, []).append(idx)

    clusters = list(cluster_map.values())
    return clusters, nli_pairs


# ──────────── Compute Semantic Entropy ──────────────────────────

def compute_semantic_entropy(
    responses: List[Dict],
    clusters: List[List[int]],
) -> float:
    """
    Compute Semantic Entropy from response log-probs and clusters.

    SE = -sum(P(C) * log(P(C)))
    where P(C) = sum(exp(mean_log_prob(r)) for r in C) / Z

    Uses length-normalized log-probs (mean_log_prob) to avoid bias
    towards shorter responses.
    """
    if len(clusters) <= 1:
        return 0.0

    # Use mean_log_prob (length-normalized) for fair comparison
    cluster_log_probs = []
    for cluster in clusters:
        member_log_probs = [responses[i]["mean_log_prob"] for i in cluster]
        cluster_log_probs.append(_logsumexp(member_log_probs))

    # Normalize: log P(C) - log Z where Z = sum of all cluster probs
    log_z = _logsumexp(cluster_log_probs)
    log_normalized = [lp - log_z for lp in cluster_log_probs]

    # Shannon entropy: -sum(p * log(p))
    se = 0.0
    for log_p in log_normalized:
        p = math.exp(log_p)
        if p > 0:
            se -= p * log_p  # log_p is already ln(p)

    return se


def _logsumexp(values: List[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        return float("-inf")
    max_val = max(values)
    if max_val == float("-inf"):
        return float("-inf")
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


# ──────────────────────────── Main ──────────────────────────────

def main():
    load_dotenv()

    # ---- Client Gemini (Vertex AI) ----
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    # ---- Load ELI5 dataset (same as main_eli5.py) ----
    from datasets import load_dataset

    print(f"Loading ELI5 dataset ({NUM_QUESTIONS} questions, streaming)...")
    ds_stream = load_dataset(
        "sentence-transformers/eli5", split="train", streaming=True
    )
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
        for idx, item in enumerate(tqdm(items, desc="Semantic Entropy", unit="q")):
            qid = item.get("q_id", f"eli5_{idx}")

            if RESUME and qid in done:
                continue

            question = item["question"]

            print(f"\n{'=' * 70}")
            print(f"  Q {idx + 1}/{len(items)}  [qid={qid}]")
            print(f"  Question: {question}")

            # ---- Step 1: Generate K responses ----
            print(f"  Generating {K_SAMPLES} responses...")
            responses = generate_k_responses(client, question, k=K_SAMPLES)

            if len(responses) < 2:
                print(f"  Only {len(responses)} responses, skipping SE.")
                out = {
                    "question_id": qid,
                    "question": question,
                    "error": f"only {len(responses)} responses generated",
                    "responses": responses,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            for i, r in enumerate(responses):
                print(f"    R{i}: meanLogP={r['mean_log_prob']:.3f}  "
                      f"sumLogP={r['log_prob_sum']:.1f}  "
                      f"({r['num_tokens']} tok)  {r['text'][:70]}...")

            # ---- Step 2: NLI clustering ----
            print(f"  Clustering {len(responses)} responses via NLI...")
            clusters, nli_pairs = cluster_responses_nli(
                client, question, responses
            )

            print(f"  Found {len(clusters)} clusters: "
                  f"{[len(c) for c in clusters]}")
            for pair in nli_pairs:
                eq_str = "≈" if pair["equivalent"] else "≠"
                if pair["equivalent"] is None:
                    eq_str = "?"
                print(f"    R{pair['i']} {eq_str} R{pair['j']}")

            # ---- Step 3: Compute Semantic Entropy ----
            se = compute_semantic_entropy(responses, clusters)
            print(f"  Semantic Entropy: {se:.4f}")

            # ---- Step 4: Build cluster details ----
            # Assign cluster_id to each response
            response_cluster_ids = [0] * len(responses)
            cluster_details = []

            # Compute cluster probabilities for output (length-normalized)
            cluster_log_probs = []
            for cluster in clusters:
                member_log_probs = [responses[i]["mean_log_prob"]
                                    for i in cluster]
                cluster_log_probs.append(_logsumexp(member_log_probs))

            log_z = _logsumexp(cluster_log_probs)

            for cid, (cluster, clp) in enumerate(
                zip(clusters, cluster_log_probs)
            ):
                prob = math.exp(clp - log_z)
                cluster_details.append({
                    "cluster_id": cid,
                    "response_indices": cluster,
                    "probability": round(prob, 6),
                })
                for resp_idx in cluster:
                    response_cluster_ids[resp_idx] = cid

            # ---- Save ----
            out = {
                "question_id": qid,
                "question": question,
                "semantic_entropy": round(se, 6),
                "num_clusters": len(clusters),
                "clusters": cluster_details,
                "responses": [
                    {
                        "text": r["text"],
                        "log_prob_sum": round(r["log_prob_sum"], 4),
                        "cluster_id": response_cluster_ids[i],
                    }
                    for i, r in enumerate(responses)
                ],
                "nli_pairs": nli_pairs,
            }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINISHED – Summary")
    print("=" * 70)
    print_summary(OUT_JSONL)


def print_summary(jsonl_path: Path):
    """Print aggregate statistics from the SE results file."""
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

    se_vals = [r["semantic_entropy"] for r in results]
    print(f"Semantic Entropy:  mean={np.mean(se_vals):.4f}  "
          f"median={np.median(se_vals):.4f}  "
          f"std={np.std(se_vals):.4f}")
    print(f"                   min={min(se_vals):.4f}  "
          f"max={max(se_vals):.4f}")

    n_clusters = [r["num_clusters"] for r in results]
    print(f"Num clusters:      mean={np.mean(n_clusters):.2f}  "
          f"min={min(n_clusters)}  max={max(n_clusters)}")

    # Distribution of cluster counts
    from collections import Counter
    cluster_dist = Counter(n_clusters)
    print("  Cluster count distribution:")
    for k in sorted(cluster_dist.keys()):
        count = cluster_dist[k]
        bar = "#" * count
        print(f"    {k} clusters: {count:3d}  {bar}")

    print()


if __name__ == "__main__":
    main()
