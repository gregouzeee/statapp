import math
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================
# Config
# =========================
MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 300
TOP_LOGPROBS = 5  # nombre d'alternatives retournées à chaque pas


def make_summary_prompt(text: str) -> str:
    return f"""
Tu es un assistant de résumé.

Ta tâche :
- Résume fidèlement le texte ci-dessous.
- Le résumé doit contenir exactement 5 phrases.
- Écris en français.
- N'ajoute aucune information absente du texte.
- Sois clair, précis et synthétique.
- Retourne uniquement le résumé final, sans titre, sans liste à puces, sans introduction.

Texte à résumer :
{text.strip()}
""".strip()


def build_client() -> genai.Client:
    load_dotenv()

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")

    if not project:
        raise RuntimeError("GCP_PROJECT not found in .env")

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


def _safe_exp(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        return math.exp(x)
    except Exception:
        return None


def _normalize_probs_from_logprobs(logprobs: list[float]) -> list[float]:
    if not logprobs:
        return []

    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    s = sum(exps)
    if s == 0:
        return []
    return [x / s for x in exps]


def _entropy_from_probs(probs: list[float]) -> float | None:
    if not probs:
        return None
    return -sum(p * math.log(p) for p in probs if p > 0)


def _extract_text_from_candidate(candidate: Any) -> str:
    try:
        parts = candidate.content.parts or []
    except Exception:
        return ""

    out = []
    for part in parts:
        txt = getattr(part, "text", None)
        if txt:
            out.append(txt)
    return "".join(out)


def _analyze_candidate_logprobs(candidate: Any) -> dict:
    """
    Analyse token par token :
    - logprob du token choisi
    - probabilité approx du token choisi
    - entropie locale sur top-k
    - marge top1-top2
    - positions incertaines
    """
    logprobs_result = getattr(candidate, "logprobs_result", None)
    avg_logprobs = getattr(candidate, "avg_logprobs", None)

    if logprobs_result is None:
        return {
            "avg_logprobs": avg_logprobs,
            "tokens": [],
            "summary_stats": {
                "n_tokens": 0,
                "mean_chosen_logprob": None,
                "mean_chosen_prob": None,
                "mean_entropy_topk": None,
                "mean_margin_top1_top2": None,
                "uncertain_token_count": 0,
                "uncertain_token_ratio": None,
            },
            "uncertain_spans": [],
        }

    chosen = getattr(logprobs_result, "chosen_candidates", None) or []
    top = getattr(logprobs_result, "top_candidates", None) or []

    n = min(len(chosen), len(top))
    token_rows = []

    for i in range(n):
        chosen_i = chosen[i]
        top_i = top[i]

        chosen_token = getattr(chosen_i, "token", None)
        chosen_logprob = getattr(chosen_i, "log_probability", None)
        chosen_prob = _safe_exp(chosen_logprob)

        candidates = getattr(top_i, "candidates", None) or []
        top_logprobs = [
            getattr(c, "log_probability", None)
            for c in candidates
            if getattr(c, "log_probability", None) is not None
        ]
        top_tokens = [getattr(c, "token", None) for c in candidates]

        probs_topk = _normalize_probs_from_logprobs(top_logprobs)
        entropy_topk = _entropy_from_probs(probs_topk)

        margin_top1_top2 = None
        top1_prob = None
        top2_prob = None
        if len(probs_topk) >= 1:
            top1_prob = probs_topk[0]
        if len(probs_topk) >= 2:
            top2_prob = probs_topk[1]
            margin_top1_top2 = top1_prob - top2_prob

        # Heuristique d'incertitude locale :
        # - token choisi avec faible probabilité
        # - ou distribution top-k assez plate
        is_uncertain = False
        if chosen_prob is not None and chosen_prob < 0.30:
            is_uncertain = True
        if entropy_topk is not None and entropy_topk > 1.00:
            is_uncertain = True
        if margin_top1_top2 is not None and margin_top1_top2 < 0.15:
            is_uncertain = True

        token_rows.append({
            "index": i,
            "token": chosen_token,
            "chosen_logprob": chosen_logprob,
            "chosen_prob": chosen_prob,
            "entropy_topk": entropy_topk,
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "margin_top1_top2": margin_top1_top2,
            "top_tokens": top_tokens,
            "top_logprobs": top_logprobs,
            "is_uncertain": is_uncertain,
        })

    def _mean(vals: list[float | None]) -> float | None:
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    uncertain_positions = [row["index"] for row in token_rows if row["is_uncertain"]]

    # Fusionne les tokens incertains contigus en segments
    uncertain_spans = []
    if uncertain_positions:
        start = uncertain_positions[0]
        prev = start
        for pos in uncertain_positions[1:]:
            if pos == prev + 1:
                prev = pos
            else:
                uncertain_spans.append((start, prev))
                start = pos
                prev = pos
        uncertain_spans.append((start, prev))

    # Reconstitue un petit extrait texte pour chaque segment
    spans_with_text = []
    for start, end in uncertain_spans:
        txt = "".join(
            row["token"] or ""
            for row in token_rows[start:end + 1]
        )
        spans_with_text.append({
            "start": start,
            "end": end,
            "text": txt,
            "length": end - start + 1,
        })

    summary_stats = {
        "n_tokens": len(token_rows),
        "mean_chosen_logprob": _mean([r["chosen_logprob"] for r in token_rows]),
        "mean_chosen_prob": _mean([r["chosen_prob"] for r in token_rows]),
        "mean_entropy_topk": _mean([r["entropy_topk"] for r in token_rows]),
        "mean_margin_top1_top2": _mean([r["margin_top1_top2"] for r in token_rows]),
        "uncertain_token_count": len(uncertain_positions),
        "uncertain_token_ratio": (
            len(uncertain_positions) / len(token_rows) if token_rows else None
        ),
    }

    return {
        "avg_logprobs": avg_logprobs,
        "tokens": token_rows,
        "summary_stats": summary_stats,
        "uncertain_spans": spans_with_text,
    }


def summarize_text_with_uncertainty(text: str) -> dict:
    client = build_client()
    prompt = make_summary_prompt(text)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            candidate_count=1,
            response_logprobs=True,
            logprobs=TOP_LOGPROBS,
        ),
    )

    summary = (response.text or "").strip()
    if not summary:
        raise RuntimeError("Empty response from Vertex AI")

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("No candidates returned by Vertex AI")

    candidate = candidates[0]
    analysis = _analyze_candidate_logprobs(candidate)

    return {
        "summary": summary,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "top_logprobs": TOP_LOGPROBS,
        "uncertainty": analysis,
    }


def summarize_text(text: str) -> str:

    result = summarize_text_with_uncertainty(text)
    return result["summary"]


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python summarize.py "Ton texte à résumer ici"')
        sys.exit(1)

    text = sys.argv[1].strip()
    if not text:
        raise ValueError("Le texte d'entrée est vide.")

    result = summarize_text_with_uncertainty(text)

    print("===== SUMMARY =====")
    print(result["summary"])

    print("\n===== UNCERTAINTY STATS =====")
    for k, v in result["uncertainty"]["summary_stats"].items():
        print(f"{k}: {v}")

    print("\n===== UNCERTAIN SPANS =====")
    for span in result["uncertainty"]["uncertain_spans"][:10]:
        print(span)


if __name__ == "__main__":
    main()