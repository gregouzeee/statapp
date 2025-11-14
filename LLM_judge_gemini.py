# selfcheck_gemini_batch.py
import os, re, json, time, logging
from typing import List, Optional
import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class SelfCheckGeminiBatch:
    """
    Compare chaque phrase à chaque passage (M x K) en batchant sur les phrases.
    Total d'appels ≈ K * ceil(M / batch_size) au lieu de M * K.

    Pour un batch (même passage, plusieurs phrases), le modèle doit retourner:
      {"answers": [true|false, ...]}  # même longueur/ordre que le batch

    Scoring:
      True  -> 0.0  (supporté par le contexte)
      False -> 1.0  (non supporté)
      None/parse KO -> 0.5 (neutre)
    """
    def __init__(
        self,
        model: str = "models/gemini-2.5-flash-lite-preview-06-17",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 64,
        request_timeout: Optional[float] = None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        logger.info(
            "Initialized SelfCheckGeminiBatch(model=%s, temp=%.2f, max_output_tokens=%d)",
            model, temperature, max_output_tokens
        )

    # ---------- prompt & parsing ----------
    def _build_prompt(self, context: str, sentences: List[str]) -> str:
        """
        Demande une réponse JSON compacte: {"answers":[true,false,...]}
        """
        lines = [
            "<ROLE> You are a strict factuality checker. </ROLE>",
            "<Instructions>",
            "Given a CONTEXT and a list of SENTENCES, return JSON exactly:",
            '{ "answers": [ <true|false>, ... ] }',
            "The array MUST match SENTENCES in length and order.",
            "true iff the sentence is supported by CONTEXT; false otherwise.",
            "Return only the JSON object. No prose, no markdown fences.",
            "</Instructions>",
            "",
            f"CONTEXT: {context.replace('\\n', ' ')}",
            "",
            "SENTENCES:",
        ]
        for i, s in enumerate(sentences):
            lines.append(f"{i+1}. {s}")
        lines.append("")
        lines.append('Return only: {"answers": [...]}')
        return "\n".join(lines)

    def _parse_answers(self, raw_text: str, expected_len: int) -> Optional[List[Optional[bool]]]:
        if not raw_text:
            logger.warning("Empty response text; returning None")
            return None

        # supprime eventuels ```json ... ```
        cleaned = re.sub(r"^(```json|```|''')\s*", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)

        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            logger.warning("No JSON object found in response: %s", raw_text[:200])
            return None

        try:
            data = json.loads(m.group(0))
            arr = data.get("answers", None)
            if not isinstance(arr, list):
                logger.warning("JSON has no 'answers' list: %s", data)
                return None

            out: List[Optional[bool]] = []
            for x in arr[:expected_len]:
                if isinstance(x, bool):
                    out.append(x)
                elif isinstance(x, str):
                    xl = x.strip().lower()
                    if xl in ("true", "yes"):
                        out.append(True)
                    elif xl in ("false", "no"):
                        out.append(False)
                    else:
                        out.append(None)
                else:
                    out.append(None)

            while len(out) < expected_len:
                out.append(None)
            return out

        except Exception as e:
            logger.exception("JSON parse error: %s", e)
            return None

    # ---------- appels ----------
    def _call_batch(self, context: str, sentences_chunk: List[str]) -> List[float]:
        prompt = self._build_prompt(context, sentences_chunk)
        t0 = time.time()
        try:
            logger.debug("Calling Gemini: model=%s, chunk_size=%d", self.model, len(sentences_chunk))
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            )
            raw = getattr(resp, "text", "") or ""
            logger.debug("Response length=%d chars", len(raw))
            parsed = self._parse_answers(raw, expected_len=len(sentences_chunk))
        except Exception as e:
            logger.exception("Gemini API error: %s", e)
            parsed = None

        dt = (time.time() - t0) * 1000.0
        logger.info("Batch call done in %.1f ms (chunk=%d)", dt, len(sentences_chunk))

        if not parsed:
            logger.warning("Parsed answers is None; returning neutral scores")
            return [0.5] * len(sentences_chunk)

        return [0.0 if v is True else 1.0 if v is False else 0.5 for v in parsed]

    # ---------- API publique ----------
    def predict_matrix(
        self,
        sentences: List[str],
        passages: List[str],
        batch_size: int = 32,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Renvoie la matrice M x K (phrase i vs passage j)
        """
        M, K = len(sentences), len(passages)
        if M == 0 or K == 0:
            return np.zeros((M, K), dtype=float)

        if verbose:
            logger.setLevel(logging.INFO)

        scores = np.zeros((M, K), dtype=float)
        for j in range(K):
            ctx = passages[j].replace("\n", " ")
            ctx_len = len(ctx)
            chunk_idx = 0
            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)
                chunk_idx += 1
                logger.info(
                    "sentence %d..%d | chunk %d | passage %d/%d (ctx_len=%d)",
                    start, end - 1, chunk_idx, j + 1, K, ctx_len
                )
                chunk = sentences[start:end]
                chunk_scores = self._call_batch(ctx, chunk)
                scores[start:end, j] = np.array(chunk_scores, dtype=float)
        return scores

    def predict_mean(
        self,
        sentences: List[str],
        passages: List[str],
        batch_size: int = 32,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Renvoie un vecteur M (moyenne des scores sur les K passages).
        """
        mat = self.predict_matrix(sentences, passages, batch_size=batch_size, verbose=verbose)
        return mat.mean(axis=1)
