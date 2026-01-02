"""
test_mmlu.py

Conformal prediction sur MMLU (cais/mmlu) avec Gemini via generate_content (AI Studio friendly).

Objectif (lisibilité + un seul fichier):
- Un seul fichier: startup_code/Conformal_prediction/output.jsonl
- Pas de lignes "summary"
- Format lisible:
    - 1 ligne "meta" au début (paramètres + explication des labels)
    - 1 ligne par exemple "example" (A/B/C/D explicites)
      * d'abord on remplit probs/true/pred
      * puis on calcule LAC/APS et on écrit le fichier final proprement (rewrite)

Reprise:
- Si output.jsonl existe déjà, on recharge les exemples déjà présents (idx) et on ne refait pas d'appels pour ces idx.
- Le script réécrit toujours un fichier final "propre" (meta + exemples), pour éviter le mélange de types.

Notes:
- labels (true) = index de la bonne réponse dans le dataset:
    0 -> A, 1 -> B, 2 -> C, 3 -> D
- probs = distribution normalisée [pA, pB, pC, pD] obtenue à partir des confiances Gemini (0-100).

"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types

from conformal_prediction import ConformalPredictor


# -----------------------
# Logging
# -----------------------

def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mmlu_conformal")
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


logger = setup_logger(logging.INFO)

LABELS_STR = ["A", "B", "C", "D"]


# -----------------------
# Env
# -----------------------

load_dotenv(Path(__file__).parent.parent.parent / ".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# -----------------------
# JSONL helpers
# -----------------------

def read_output_jsonl(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Lit output.jsonl et renvoie:
      - meta (dict) si trouvé
      - examples_by_idx: dict idx -> example dict (type="example")

    Si des lignes sont corrompues, elles sont ignorées.
    Si plusieurs meta, on prend la dernière.
    Si plusieurs examples du même idx, on prend la dernière.
    """
    meta = None
    examples: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return meta, examples

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            t = obj.get("type")
            if t == "meta":
                meta = obj
            elif t == "example":
                try:
                    idx = int(obj["idx"])
                except Exception:
                    continue
                examples[idx] = obj

    return meta, examples


def write_output_jsonl(path: Path, meta: Dict[str, Any], examples_by_idx: Dict[int, Dict[str, Any]]) -> None:
    """
    Réécrit le fichier en entier (format propre):
      1 ligne meta
      puis examples triés par idx
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for idx in sorted(examples_by_idx.keys()):
            f.write(json.dumps(examples_by_idx[idx], ensure_ascii=False) + "\n")


# -----------------------
# Gemini batching-in-prompt
# -----------------------

class GeminiMMLUBatched:
    """
    1 appel generate_content = JSON:
      {"items":[{"i":0,"A":..,"B":..,"C":..,"D":..}, ...]}
    puis normalisation -> probas.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        logger.info("Gemini init | model=%s | temp=%.2f", model, temperature)

    def _build_prompt_batch(self, questions: List[str], choices_list: List[List[str]]) -> str:
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

    @staticmethod
    def _clean(raw: str) -> str:
        cleaned = re.sub(r"^(```json|```|''')\s*", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)
        return cleaned.strip()

    def _parse_batch(self, raw_text: str, batch_len: int) -> List[np.ndarray]:
        if not raw_text:
            return [np.ones(4) / 4 for _ in range(batch_len)]

        cleaned = self._clean(raw_text)
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

                vec = np.array([
                    float(it.get("A", 0)),
                    float(it.get("B", 0)),
                    float(it.get("C", 0)),
                    float(it.get("D", 0)),
                ], dtype=float)

                if vec.sum() > 0:
                    vec = vec / vec.sum()
                else:
                    vec = np.ones(4) / 4

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

    def call_with_retry(
        self,
        questions: List[str],
        choices_list: List[List[str]],
        max_retries: int = 5,
        base_sleep_s: float = 5.0,
    ) -> List[np.ndarray]:
        prompt = self._build_prompt_batch(questions, choices_list)
        batch_len = len(questions)

        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                dt = (time.time() - t0) * 1000.0
                raw = getattr(resp, "text", "") or ""
                logger.info("generate_content OK | batch_len=%d | %.1f ms", batch_len, dt)
                return self._parse_batch(raw, batch_len)

            except Exception as e:
                msg = str(e)
                is_429 = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("Too Many Requests" in msg)
                logger.warning(
                    "API error attempt %d/%d | 429=%s | %s",
                    attempt + 1, max_retries + 1, is_429, msg[:200]
                )

                if attempt >= max_retries:
                    logger.error("Max retries -> fallback uniform for this batch.")
                    return [np.ones(4) / 4 for _ in range(batch_len)]

                # Backoff exponentiel + petit jitter
                sleep_s = base_sleep_s * (2 ** attempt) + 0.2 * attempt
                logger.info("Sleeping %.1fs then retry...", sleep_s)
                time.sleep(sleep_s)

        return [np.ones(4) / 4 for _ in range(batch_len)]


# -----------------------
# Conformal helpers
# -----------------------

def compute_conformal_for_examples(
    probs_mat: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    split_cal_ratio: float,
) -> Tuple[ConformalPredictor, ConformalPredictor, int]:
    """
    Calibre LAC & APS sur les premiers n_cal exemples,
    renvoie (cp_lac, cp_aps, n_cal).
    """
    n_total = len(labels)
    n_cal = int(n_total * split_cal_ratio)
    if n_cal <= 0 or n_cal >= n_total:
        raise ValueError("split_cal_ratio invalide.")

    probs_cal = probs_mat[:n_cal]
    y_cal = labels[:n_cal]

    cp_lac = ConformalPredictor(score_fn="lac", alpha=alpha).calibrate(probs_cal, y_cal)
    cp_aps = ConformalPredictor(score_fn="aps", alpha=alpha).calibrate(probs_cal, y_cal)
    return cp_lac, cp_aps, n_cal


def as_probs_dict(p: np.ndarray) -> Dict[str, float]:
    return {"A": float(p[0]), "B": float(p[1]), "C": float(p[2]), "D": float(p[3])}


# -----------------------
# Main
# -----------------------

def main():
    # ---- Config expérience
    subject = "high_school_mathematics"
    n_samples = 50
    alpha = 0.1
    split_cal_ratio = 0.5

    # Modèle (tu as dit que celui-ci marche chez toi)
    model = "models/gemini-2.5-flash-lite"
    temperature = 0.0

    # Throttling / batching (dans le prompt)
    batch_size = 15
    batches_per_group = 15
    sleep_between_groups_s = 60

    # Output unique
    out_jsonl = Path("statap_code") / "Conformal_prediction" / "output.jsonl"

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY introuvable (.env).")

    logger.info("START | subject=%s | n_samples=%d | alpha=%.2f", subject, n_samples, alpha)
    logger.info("model=%s | batch_size=%d | group=%d | sleep=%ds", model, batch_size, batches_per_group, sleep_between_groups_s)
    logger.info("output=%s", out_jsonl.as_posix())

    # ---- Charger dataset
    dataset = load_dataset("cais/mmlu", subject, split="test")
    if len(dataset) > n_samples:
        dataset = dataset.select(range(n_samples))
    n_total = len(dataset)

    questions = [ex["question"] for ex in dataset]
    choices_list = [ex["choices"] for ex in dataset]
    true_labels = np.array([ex["answer"] for ex in dataset], dtype=int)  # 0=A,1=B,2=C,3=D

    # ---- Charger fichier existant (reprise)
    old_meta, examples_by_idx = read_output_jsonl(out_jsonl)

    # ---- Meta (on garde des infos utiles + explication labels)
    meta = {
        "type": "meta",
        "created_at": old_meta.get("created_at") if isinstance(old_meta, dict) and old_meta.get("created_at") else time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject,
        "n_samples": int(n_samples),
        "alpha": float(alpha),
        "split_cal_ratio": float(split_cal_ratio),
        "model": model,
        "temperature": float(temperature),
        "batch_size": int(batch_size),
        "batches_per_group": int(batches_per_group),
        "sleep_between_groups_s": int(sleep_between_groups_s),
        "label_meaning": "true is the correct option letter (A/B/C/D) from the dataset. Internally answer is an index: 0=A,1=B,2=C,3=D.",
        # q_hat seront remplis après calibration
        "q_hat_lac": None,
        "q_hat_aps": None,
    }

    # ---- Gemini client
    gemini = GeminiMMLUBatched(model=model, api_key=GEMINI_API_KEY, temperature=temperature, max_output_tokens=2048)

    # ---- Appels Gemini manquants
    done = set(examples_by_idx.keys())
    logger.info("Resume: %d/%d examples already present", len(done), n_total)

    calls_in_group = 0
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        idxs = list(range(start, end))
        idxs_todo = [i for i in idxs if i not in done]
        if not idxs_todo:
            continue

        logger.info("Call | items %d-%d/%d | todo=%d", start + 1, end, n_total, len(idxs_todo))
        q_batch = [questions[i] for i in idxs_todo]
        c_batch = [choices_list[i] for i in idxs_todo]

        probs_list = gemini.call_with_retry(q_batch, c_batch, max_retries=5, base_sleep_s=5.0)

        for idx, p in zip(idxs_todo, probs_list):
            y = int(true_labels[idx])
            pred_idx = int(np.argmax(p))
            examples_by_idx[int(idx)] = {
                "type": "example",
                "idx": int(idx),
                # split et conformal seront remplis à la finalisation
                "split": None,
                "true": LABELS_STR[y],
                "pred": LABELS_STR[pred_idx],
                "probs": as_probs_dict(p),
                # Conformal (sera rempli ensuite)
                "lac": None,
                "aps": None,
            }
            done.add(idx)

        calls_in_group += 1
        if calls_in_group >= batches_per_group and len(done) < n_total:
            logger.info("Sleep %ds after %d calls...", sleep_between_groups_s, batches_per_group)
            time.sleep(sleep_between_groups_s)
            calls_in_group = 0

        # petite sauvegarde intermédiaire (en réécrivant proprement)
        write_output_jsonl(out_jsonl, meta=meta, examples_by_idx=examples_by_idx)
        logger.info("Saved intermediate (rewrite) | done=%d/%d", len(done), n_total)

    # ---- Construire matrice probs + labels (ordre 0..n_total-1)
    probs_mat = np.full((n_total, 4), np.nan, dtype=float)
    labels = np.full((n_total,), -1, dtype=int)

    for i in range(n_total):
        ex = examples_by_idx.get(i)
        if not ex:
            continue
        p = ex.get("probs", {})
        if isinstance(p, dict) and all(k in p for k in ("A", "B", "C", "D")):
            probs_mat[i] = np.array([p["A"], p["B"], p["C"], p["D"]], dtype=float)
        # label vrai depuis dataset (source de vérité)
        labels[i] = int(true_labels[i])

    # Fill missing probs with uniform (sécurité)
    missing = int(np.isnan(probs_mat).any(axis=1).sum())
    if missing > 0:
        logger.warning("Missing probs for %d items -> set uniform", missing)
        for i in range(n_total):
            if np.isnan(probs_mat[i]).any():
                probs_mat[i] = np.ones(4) / 4
                # si exemple manquant, on le crée
                if i not in examples_by_idx:
                    y = int(true_labels[i])
                    examples_by_idx[i] = {
                        "type": "example",
                        "idx": int(i),
                        "split": None,
                        "true": LABELS_STR[y],
                        "pred": "A",  # argmax uniforme arbitraire
                        "probs": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                        "lac": None,
                        "aps": None,
                    }

    # ---- Calibration conformal
    cp_lac, cp_aps, n_cal = compute_conformal_for_examples(
        probs_mat=probs_mat,
        labels=labels,
        alpha=alpha,
        split_cal_ratio=split_cal_ratio,
    )
    meta["q_hat_lac"] = float(cp_lac.q_hat)
    meta["q_hat_aps"] = float(cp_aps.q_hat)
    meta["n_total"] = int(n_total)
    meta["n_cal"] = int(n_cal)
    meta["n_test"] = int(n_total - n_cal)

    # ---- Remplir split + ensembles conformes (pour TOUS les exemples, pas seulement test)
    for i in range(n_total):
        p = probs_mat[i]
        y = int(labels[i])

        split = "cal" if i < n_cal else "test"
        pred_idx = int(np.argmax(p))

        C_lac = cp_lac.predict(p)
        C_aps = cp_aps.predict(p)

        examples_by_idx[i]["split"] = split
        examples_by_idx[i]["true"] = LABELS_STR[y]
        examples_by_idx[i]["pred"] = LABELS_STR[pred_idx]

        examples_by_idx[i]["lac"] = {
            "set": [LABELS_STR[int(k)] for k in C_lac],
            "covered": bool(y in C_lac),
            "size": int(len(C_lac)),
            "p_true": float(p[y]),
        }
        examples_by_idx[i]["aps"] = {
            "set": [LABELS_STR[int(k)] for k in C_aps],
            "covered": bool(y in C_aps),
            "size": int(len(C_aps)),
            "p_true": float(p[y]),
        }

    # ---- Écrire fichier final (meta + examples) - aucune ligne summary
    write_output_jsonl(out_jsonl, meta=meta, examples_by_idx=examples_by_idx)
    logger.info("DONE. Clean output written (no summary): %s", out_jsonl.as_posix())


if __name__ == "__main__":
    main()
