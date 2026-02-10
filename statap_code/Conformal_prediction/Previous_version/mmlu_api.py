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

LABELS_STR = ["A", "B", "C", "D"]


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mmlu_api")
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


logger = setup_logger(logging.INFO)


def read_output_jsonl(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for idx in sorted(examples_by_idx.keys()):
            f.write(json.dumps(examples_by_idx[idx], ensure_ascii=False) + "\n")


class GeminiMMLUBatched:
    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_output_tokens: int = 2048):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

    def _build_prompt_batch(self, questions: List[str], choices_list: List[List[str]]) -> str:
        lines = [
            "<ROLE>You are an exam solver.</ROLE>",
            "<INSTRUCTIONS>",
            "For EACH question below, provide confidence scores (0-100) for options A,B,C,D.",
            "Return ONLY a valid JSON object with schema:",
            '{"items":[{"i":0,"A":75,"B":10,"C":10,"D":5}, ...]}',
            "Rules: include exactly one object per question; values are numbers; no text besides JSON.",
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

                vec = vec / vec.sum() if vec.sum() > 0 else (np.ones(4) / 4)
                out_map[i] = vec

            out, missing = [], 0
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
                logger.warning("API error attempt %d/%d | 429=%s | %s", attempt + 1, max_retries + 1, is_429, msg[:200])

                if attempt >= max_retries:
                    logger.error("Max retries -> fallback uniform for this batch.")
                    return [np.ones(4) / 4 for _ in range(batch_len)]

                sleep_s = base_sleep_s * (2 ** attempt) + 0.2 * attempt
                logger.info("Sleeping %.1fs then retry...", sleep_s)
                time.sleep(sleep_s)

        return [np.ones(4) / 4 for _ in range(batch_len)]


def as_probs_dict(p: np.ndarray) -> Dict[str, float]:
    return {"A": float(p[0]), "B": float(p[1]), "C": float(p[2]), "D": float(p[3])}


def collect_mmlu_probs(
    *,
    subject: str,
    n_samples: int,
    out_jsonl: Path,
    model: str,
    temperature: float,
    batch_size: int,
    batches_per_group: int,
    sleep_between_groups_s: int,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """
    1) Charge MMLU
    2) Remplit output.jsonl progressivement avec probs
    3) Retourne probs_mat (n,4), labels (n,), meta, examples_by_idx
    """
    # env
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY introuvable (.env).")

    dataset = load_dataset("cais/mmlu", subject, split="test")
    if len(dataset) > n_samples:
        dataset = dataset.select(range(n_samples))
    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError("Empty dataset.")

    questions = [ex["question"] for ex in dataset]
    choices_list = [ex["choices"] for ex in dataset]
    true_labels = np.array([ex["answer"] for ex in dataset], dtype=int)  # 0..3

    old_meta, examples_by_idx = read_output_jsonl(out_jsonl)

    meta = {
        "type": "meta",
        "created_at": old_meta.get("created_at") if isinstance(old_meta, dict) and old_meta.get("created_at") else time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "subject": subject,
        "n_samples": int(n_samples),
        "model": model,
        "temperature": float(temperature),
        "batch_size": int(batch_size),
        "batches_per_group": int(batches_per_group),
        "sleep_between_groups_s": int(sleep_between_groups_s),
        "label_meaning": "answer is index 0=A,1=B,2=C,3=D.",
        "random_seed": int(random_seed),
    }

    gemini = GeminiMMLUBatched(model=model, api_key=api_key, temperature=temperature, max_output_tokens=2048)

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
                "split": None,
                "true": LABELS_STR[y],
                "pred": LABELS_STR[pred_idx],
                "probs": as_probs_dict(p),
                "lac": None,
                "aps": None,
            }
            done.add(idx)

        calls_in_group += 1
        if calls_in_group >= batches_per_group and len(done) < n_total:
            logger.info("Sleep %ds after %d calls...", sleep_between_groups_s, batches_per_group)
            time.sleep(sleep_between_groups_s)
            calls_in_group = 0

        write_output_jsonl(out_jsonl, meta=meta, examples_by_idx=examples_by_idx)
        logger.info("Saved intermediate | done=%d/%d", len(done), n_total)

    # build probs_mat
    probs_mat = np.full((n_total, 4), np.nan, dtype=float)
    labels = true_labels.copy()

    for i in range(n_total):
        ex = examples_by_idx.get(i)
        if ex and isinstance(ex.get("probs"), dict):
            p = ex["probs"]
            if all(k in p for k in ("A", "B", "C", "D")):
                probs_mat[i] = np.array([p["A"], p["B"], p["C"], p["D"]], dtype=float)

    missing_mask = np.isnan(probs_mat).any(axis=1)
    if int(missing_mask.sum()) > 0:
        logger.warning("Missing probs for %d items -> set uniform", int(missing_mask.sum()))
        probs_mat[missing_mask] = np.ones(4) / 4

    return probs_mat, labels, meta, examples_by_idx
