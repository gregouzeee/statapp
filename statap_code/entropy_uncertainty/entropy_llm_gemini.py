import json
import logging
import re
from typing import Dict, List

import numpy as np
from google import genai
from google.genai import types

try:
    from .entropy_uncertainty import confidence_from_entropy, normalize_probs, normalized_entropy
except ImportError:  # Allow running as a script without package context
    from entropy_uncertainty import confidence_from_entropy, normalize_probs, normalized_entropy

logger = logging.getLogger(__name__)


def _clean_json(raw: str) -> str:
    cleaned = re.sub(r"^(```json|```|''')\s*", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()


class GeminiMCQEntropy:
    """
    Ask Gemini for confidence scores over A/B/C/D, normalize them, then
    compute entropy-based uncertainty.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        logger.info("GeminiMCQEntropy init | model=%s | temp=%.2f", model, temperature)

    def _build_prompt(self, questions: List[str], choices_list: List[List[str]]) -> str:
        lines = [
            "<ROLE>You are an exam solver.</ROLE>",
            "<INSTRUCTIONS>",
            "For EACH question below, provide confidence scores (0-100) for options A,B,C,D.",
            "Return ONLY a valid JSON object with the exact schema:",
            '{"items":[{"i":0,"A":75,"B":10,"C":10,"D":5}, ...]}',
            "Rules:",
            "- 'i' is the index of the question in this batch (starting at 0).",
            "- Include exactly one object per question.",
            "- Values must be numbers.",
            "- No explanations, no markdown fences, only JSON.",
            "</INSTRUCTIONS>",
            "",
            "BATCH:",
        ]
        for i, (q, c) in enumerate(zip(questions, choices_list)):
            lines.append(f"Question #{i}: {q}")
            lines.append(f"A. {c[0]}")
            lines.append(f"B. {c[1]}")
            lines.append(f"C. {c[2]}")
            lines.append(f"D. {c[3]}")
            lines.append("")
        lines.append('Return ONLY: {"items":[...]}')
        return "\n".join(lines)

    def _parse_probs(self, raw_text: str, batch_len: int) -> List[np.ndarray]:
        if not raw_text:
            return [np.ones(4) / 4 for _ in range(batch_len)]

        cleaned = _clean_json(raw_text)
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            logger.warning("No JSON found; fallback uniform.")
            return [np.ones(4) / 4 for _ in range(batch_len)]

        try:
            data = json.loads(m.group(0))
            items = data.get("items")
            if not isinstance(items, list):
                logger.warning("JSON missing items list; fallback uniform.")
                return [np.ones(4) / 4 for _ in range(batch_len)]

            out_map: Dict[int, np.ndarray] = {}
            for it in items:
                if not isinstance(it, dict) or "i" not in it:
                    continue
                try:
                    i = int(it["i"])
                except Exception:
                    continue
                if not (0 <= i < batch_len):
                    continue

                vec = np.array(
                    [
                        float(it.get("A", 0)),
                        float(it.get("B", 0)),
                        float(it.get("C", 0)),
                        float(it.get("D", 0)),
                    ],
                    dtype=float,
                )
                vec = normalize_probs(vec)
                out_map[i] = vec

            out: List[np.ndarray] = []
            missing = 0
            for i in range(batch_len):
                if i in out_map:
                    out.append(out_map[i])
                else:
                    missing += 1
                    out.append(np.ones(4) / 4)
            if missing:
                logger.warning("Missing %d/%d items in parse; uniform fallback for missing.", missing, batch_len)
            return out
        except Exception:
            logger.exception("Parse error; fallback uniform.")
            return [np.ones(4) / 4 for _ in range(batch_len)]

    def predict(
        self,
        questions: List[str],
        choices_list: List[List[str]],
    ) -> List[Dict[str, object]]:
        prompt = self._build_prompt(questions, choices_list)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        raw = getattr(resp, "text", "") or ""
        probs_list = self._parse_probs(raw, len(questions))

        results = []
        for p in probs_list:
            h_norm = float(normalized_entropy(p))
            conf = float(confidence_from_entropy(p))
            pred_idx = int(np.argmax(p))
            results.append(
                {
                    "probs": {"A": float(p[0]), "B": float(p[1]), "C": float(p[2]), "D": float(p[3])},
                    "pred": ["A", "B", "C", "D"][pred_idx],
                    "entropy_norm": h_norm,
                    "confidence": conf,
                }
            )
        return results
