"""
WEPR — Weighted Entropy Production Rate
========================================
Implementation of EPR and WEPR from:
    Moslonka et al., "Learned Hallucination Detection in Black-Box LLMs
    using Token-level Entropy Production Rate" (arXiv:2509.04492v2, 2026)

Functions:
    - entropic_contributions(token_data, K) -> array (L x K)
    - epr(token_data, K) -> float
    - wepr_features(token_data, K) -> dict of features for logistic regression
    - wepr_score(token_data, K, beta, gamma) -> float
    - wepr_token_scores(token_data, K, beta) -> list of floats (per-token)
    - wepr_word_scores(token_data, K, beta) -> list of floats (per-word)
"""

import math
import string
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────── Core computations ────────────────────────

def entropic_contributions(token_data: List[Dict], K: int) -> np.ndarray:
    """
    Compute the entropic contribution s(k,j) = -p(k,j) * log2(p(k,j))
    for each token position j and rank k.

    Args:
        token_data: list of dicts with "top_k" field containing
                    [{"token": str, "log_prob": float}, ...]
        K: number of top ranks to use

    Returns:
        np.ndarray of shape (L, K) where L = len(token_data)
        s[j, k] = entropic contribution of rank k at position j
    """
    L = len(token_data)
    s = np.zeros((L, K))

    for j, td in enumerate(token_data):
        top_k = td.get("top_k", [])
        # Sort by descending probability (top_k should already be sorted,
        # but let's be safe)
        top_k_sorted = sorted(top_k, key=lambda x: x["log_prob"], reverse=True)

        for k in range(min(K, len(top_k_sorted))):
            log_p = top_k_sorted[k]["log_prob"]
            p = math.exp(log_p)
            if p > 0:
                s[j, k] = -p * math.log2(p)

    return s


def per_token_entropy(token_data: List[Dict], K: int) -> np.ndarray:
    """
    Compute truncated entropy H_K(j) at each position j.
    H_K(j) = sum_k s(k,j) = sum over top-K of -p * log2(p)

    Returns: np.ndarray of shape (L,)
    """
    s = entropic_contributions(token_data, K)
    return s.sum(axis=1)


def epr(token_data: List[Dict], K: int) -> float:
    """
    Entropy Production Rate (Eq. 6 of the paper):
    EPR = (1/L) * sum_j H_K(j)

    Unsupervised baseline score for the whole sequence.
    """
    H = per_token_entropy(token_data, K)
    if len(H) == 0:
        return 0.0
    return float(H.mean())


# ──────────────────────── WEPR features ────────────────────────

def wepr_features(token_data: List[Dict], K: int) -> Dict:
    """
    Extract the feature vector used for WEPR logistic regression.

    Features (Eq. 8 of the paper):
        - mean_s[k]: mean of s(k,j) over all positions, for each rank k (K values)
        - max_s[k]:  max of s(k,j) over all positions, for each rank k  (K values)

    Total: 2*K features.

    Returns:
        dict with keys:
            "mean_by_rank": np.ndarray of shape (K,)
            "max_by_rank":  np.ndarray of shape (K,)
            "feature_vector": np.ndarray of shape (2*K,) — concatenation
            "s_matrix": np.ndarray of shape (L, K) — full matrix for token-level scoring
    """
    s = entropic_contributions(token_data, K)
    L = s.shape[0]

    if L == 0:
        return {
            "mean_by_rank": np.zeros(K),
            "max_by_rank": np.zeros(K),
            "feature_vector": np.zeros(2 * K),
            "s_matrix": s,
        }

    mean_by_rank = s.mean(axis=0)  # shape (K,)
    max_by_rank = s.max(axis=0)    # shape (K,)

    feature_vector = np.concatenate([mean_by_rank, max_by_rank])

    return {
        "mean_by_rank": mean_by_rank,
        "max_by_rank": max_by_rank,
        "feature_vector": feature_vector,
        "s_matrix": s,
    }


# ──────────────────────── WEPR scoring ────────────────────────

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def wepr_score(token_data: List[Dict], K: int,
               beta: np.ndarray, gamma: np.ndarray) -> float:
    """
    Compute the WEPR score for a full sequence (Eq. 8).

    WEPR = (1/L) * sum_j S_beta(j) + sum_k gamma_k * max_j s(k,j)

    where S_beta(j) = beta_0 + sum_k beta_{k+1} * s(k,j)

    Args:
        beta: np.ndarray of shape (K+1,) — [beta_0, beta_1, ..., beta_K]
        gamma: np.ndarray of shape (K,) — [gamma_1, ..., gamma_K]

    Returns:
        float: raw WEPR score (apply sigmoid for probability)
    """
    s = entropic_contributions(token_data, K)
    L = s.shape[0]
    if L == 0:
        return 0.0

    # S_beta(j) for each token
    S_beta = beta[0] + s @ beta[1:]  # shape (L,)

    # Mean + max terms
    mean_term = S_beta.mean()
    max_term = (gamma * s.max(axis=0)).sum()

    return float(mean_term + max_term)


def wepr_sequence_confidence(token_data: List[Dict], K: int,
                              beta: np.ndarray, gamma: np.ndarray) -> float:
    """
    Confidence score for the whole sequence:
    sigma(WEPR) in [0, 1]

    Higher = more likely correct (non-hallucinated).
    """
    score = wepr_score(token_data, K, beta, gamma)
    return sigmoid(score)


# ──────────────────────── Token-level scores ────────────────────────

def wepr_token_scores(token_data: List[Dict], K: int,
                      beta: np.ndarray) -> List[Dict]:
    """
    Compute per-token hallucination risk score (Eq. 10):
    score(j) = sigma(S_beta(j))

    Higher = more confident (less likely hallucinated).
    Lower = higher risk.

    Returns list of {"token": str, "confidence": float, "risk": float}
    """
    s = entropic_contributions(token_data, K)
    L = s.shape[0]

    results = []
    for j in range(L):
        S_beta_j = beta[0] + np.dot(beta[1:], s[j])
        conf = sigmoid(float(S_beta_j))
        results.append({
            "token": token_data[j]["token"],
            "confidence": conf,
            "risk": 1.0 - conf,
            "entropy": float(s[j].sum()),
        })
    return results


# ──────────────────────── Word-level scores ────────────────────────

def group_tokens_into_words(token_data: List[Dict]) -> List[Dict]:
    """
    Group sub-tokens into words (same logic as main_triviaqa.py).
    A token starting with a space begins a new word.
    """
    if not token_data:
        return []
    words = []
    current = []
    for t in token_data:
        if t["token"].startswith(" ") or not current:
            if current:
                words.append(_build_word(current))
            current = [t]
        else:
            current.append(t)
    if current:
        words.append(_build_word(current))
    return words


def _build_word(tokens: List[Dict]) -> Dict:
    word_str = "".join(t["token"] for t in tokens).strip()
    return {
        "word": word_str,
        "tokens": tokens,
        "token_indices": list(range(len(tokens))),  # will be set properly below
    }


def wepr_word_scores(token_data: List[Dict], K: int,
                     beta: np.ndarray) -> List[Dict]:
    """
    Aggregate token-level WEPR scores at word level.

    For each word:
        - mean_confidence: average of token confidences
        - min_confidence: minimum token confidence (worst token)
        - mean_risk: average risk
        - max_risk: maximum risk (worst token)

    Skips punctuation-only words.
    """
    token_scores = wepr_token_scores(token_data, K, beta)
    words = group_tokens_into_words(token_data)

    # Map tokens to word indices
    idx = 0
    results = []
    for w in words:
        n_tokens = len(w["tokens"])
        word_token_scores = token_scores[idx:idx + n_tokens]
        idx += n_tokens

        clean = w["word"].strip()
        if not clean or all(c in string.punctuation for c in clean):
            continue

        confidences = [ts["confidence"] for ts in word_token_scores]
        risks = [ts["risk"] for ts in word_token_scores]
        entropies = [ts["entropy"] for ts in word_token_scores]

        results.append({
            "word": w["word"],
            "sub_tokens": [t["token"] for t in w["tokens"]],
            "mean_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "mean_risk": float(np.mean(risks)),
            "max_risk": float(np.max(risks)),
            "mean_entropy": float(np.mean(entropies)),
            "max_entropy": float(np.max(entropies)),
        })

    return results


# ──────────────────────── Cosine leave-one-out ────────────────────────
# (reproduced from main_triviaqa.py for combination with WEPR)

def _topk_vector(top_k: List[Dict], vocab_index: Dict[str, int]) -> np.ndarray:
    vec = np.full(len(vocab_index), -50.0)
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


def cosine_leave_one_out(token_data: List[Dict]) -> List[Dict]:
    """
    Cosine-similarity importance for each token position.
    importance(t) = 1 - cos(full_vec, full_vec_without_t)
    """
    if len(token_data) <= 1:
        return [{"token": t["token"], "importance": 0.0} for t in token_data]

    vocab = set()
    for t in token_data:
        for c in t.get("top_k", []):
            vocab.add(c["token"])
    vocab_index = {tok: i for i, tok in enumerate(sorted(vocab))}

    if not vocab_index:
        return [{"token": t["token"], "importance": 0.0} for t in token_data]

    position_vecs = [_topk_vector(t.get("top_k", []), vocab_index) for t in token_data]
    full_vec = np.mean(position_vecs, axis=0)

    results = []
    for i, t in enumerate(token_data):
        if t["token"].strip() == "" or all(c in string.punctuation for c in t["token"].strip()):
            results.append({"token": t["token"], "importance": 0.0})
            continue
        remaining = [v for j, v in enumerate(position_vecs) if j != i]
        if remaining:
            without_vec = np.mean(remaining, axis=0)
        else:
            without_vec = np.zeros_like(full_vec)
        sim = _cosine_sim(full_vec, without_vec)
        results.append({
            "token": t["token"],
            "importance": 1.0 - sim,
        })
    return results


def cosine_word_leave_one_out(token_data: List[Dict]) -> List[Dict]:
    """
    Word-level cosine leave-one-out importance.
    Removes all tokens of a word at once.
    """
    words = group_tokens_into_words(token_data)

    if len(token_data) <= 1:
        return [{"word": w["word"], "importance": 0.0} for w in words]

    vocab = set()
    for t in token_data:
        for c in t.get("top_k", []):
            vocab.add(c["token"])
    vocab_index = {tok: i for i, tok in enumerate(sorted(vocab))}

    if not vocab_index:
        return [{"word": w["word"], "importance": 0.0} for w in words]

    position_vecs = [_topk_vector(t.get("top_k", []), vocab_index) for t in token_data]
    full_vec = np.mean(position_vecs, axis=0)

    # Map token positions to word indices
    token_to_word = []
    for wi, w in enumerate(words):
        for _ in w["tokens"]:
            token_to_word.append(wi)

    results = []
    for wi, w in enumerate(words):
        clean = w["word"].strip()
        if not clean or all(c in string.punctuation for c in clean):
            continue
        remaining = [v for j, v in enumerate(position_vecs) if token_to_word[j] != wi]
        if remaining:
            without_vec = np.mean(remaining, axis=0)
        else:
            without_vec = np.zeros_like(full_vec)
        sim = _cosine_sim(full_vec, without_vec)
        results.append({
            "word": w["word"],
            "sub_tokens": [t["token"] for t in w["tokens"]],
            "importance": 1.0 - sim,
        })
    return results
