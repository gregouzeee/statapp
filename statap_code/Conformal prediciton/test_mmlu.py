"""
Test de la prédiction conforme sur MMLU (Massive Multitask Language Understanding).
Utilise Gemini via l'API Google GenAI.
"""

import numpy as np
import json
import re
import time
import os
from pathlib import Path
from datasets import load_dataset
from google import genai
from google.genai import types
from typing import List, Optional
from dotenv import load_dotenv
from conformal_prediction import ConformalPredictor

# Charger le .env depuis la racine du projet
load_dotenv(Path(__file__).parent.parent.parent / ".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class GeminiMMLU:
    """
    Classe pour interroger Gemini sur des questions MMLU
    et extraire les probabilités pour chaque choix.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        print(f"GeminiMMLU initialisé avec {model}")

    def _build_prompt(self, question: str, choices: List[str]) -> str:
        """Construit le prompt pour une question MMLU."""
        prompt = f"""Question: {question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Give your confidence (0-100) for each answer option.
Return ONLY a JSON object like: {{"A": 75, "B": 10, "C": 10, "D": 5}}
No explanation, just the JSON."""
        return prompt

    def _parse_probs(self, raw_text: str) -> Optional[np.ndarray]:
        """Parse la réponse JSON et retourne les probabilités normalisées."""
        if not raw_text:
            return None

        # Nettoyer le texte
        cleaned = re.sub(r"^```json\s*", "", raw_text.strip())
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Chercher le JSON
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None

        try:
            data = json.loads(m.group(0))
            probs = np.array([
                float(data.get("A", 0)),
                float(data.get("B", 0)),
                float(data.get("C", 0)),
                float(data.get("D", 0))
            ])
            # Normaliser pour avoir une distribution de probabilités
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(4) / 4
            return probs
        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        """
        Obtient les probabilités pour chaque choix.

        Returns:
            np.array de shape (4,) avec les probabilités pour A, B, C, D
        """
        prompt = self._build_prompt(question, choices)

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=100,
                )
            )
            raw = getattr(resp, "text", "") or ""
            probs = self._parse_probs(raw)

            if probs is None:
                print(f"Warning: Could not parse response, using uniform")
                return np.ones(4) / 4

            return probs

        except Exception as e:
            print(f"API error: {e}")
            return np.ones(4) / 4

    def get_probs_batch(
        self,
        questions: List[str],
        choices_list: List[List[str]],
        delay: float = 0.1
    ) -> np.ndarray:
        """
        Obtient les probabilités pour un batch de questions.

        Args:
            questions: Liste de questions
            choices_list: Liste de listes de choix
            delay: Délai entre les appels (rate limiting)

        Returns:
            np.array de shape (n, 4)
        """
        all_probs = []
        for i, (q, c) in enumerate(zip(questions, choices_list)):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(questions)}")
            probs = self.get_probs(q, c)
            all_probs.append(probs)
            time.sleep(delay)
        return np.array(all_probs)


def main():
    # Configuration
    subject = "high_school_mathematics"
    n_samples = 50  
    alpha = 0.1

    print("=" * 60)
    print("CONFORMAL PREDICTION SUR MMLU AVEC GEMINI")
    print("=" * 60)
    print(f"Sujet: {subject}")
    print(f"Modèle: Gemini 1.5 Flash")
    print(f"Alpha: {alpha} (couverture cible: {100*(1-alpha):.0f}%)")

    # Charger le dataset
    print("\nChargement du dataset MMLU...")
    dataset = load_dataset("cais/mmlu", subject, split="test")

    if len(dataset) > n_samples:
        dataset = dataset.select(range(n_samples))
    print(f"Nombre d'exemples: {len(dataset)}")

    # Initialiser Gemini
    print("\nInitialisation de Gemini...")
    gemini = GeminiMMLU()

    # Préparer les données
    questions = [ex["question"] for ex in dataset]
    choices_list = [ex["choices"] for ex in dataset]
    labels = np.array([ex["answer"] for ex in dataset])

    # Extraire les probabilités
    print("\nExtraction des probabilités...")
    all_probs = gemini.get_probs_batch(questions, choices_list, delay=0.2)

    # Split calibration / test (50/50)
    n = len(all_probs)
    n_cal = n // 2

    probs_cal, y_cal = all_probs[:n_cal], labels[:n_cal]
    probs_test, y_test = all_probs[n_cal:], labels[n_cal:]

    print(f"\nSplit: {n_cal} calibration, {n - n_cal} test")

    # Conformal Prediction - LAC
    print("\n" + "=" * 60)
    print("RÉSULTATS LAC")
    print("=" * 60)

    cp_lac = ConformalPredictor(score_fn="lac", alpha=alpha)
    cp_lac.calibrate(probs_cal, y_cal)
    print(f"q̂ = {cp_lac.q_hat:.4f}")

    metrics_lac = cp_lac.evaluate(probs_test, y_test)
    print(f"Accuracy:      {metrics_lac['accuracy']:.2%}")
    print(f"Set Size (SS): {metrics_lac['set_size']:.2f}")
    print(f"Coverage (CR): {metrics_lac['coverage_rate']:.2%}")

    # Conformal Prediction - APS
    print("\n" + "=" * 60)
    print("RÉSULTATS APS")
    print("=" * 60)

    cp_aps = ConformalPredictor(score_fn="aps", alpha=alpha)
    cp_aps.calibrate(probs_cal, y_cal)
    print(f"q̂ = {cp_aps.q_hat:.4f}")

    metrics_aps = cp_aps.evaluate(probs_test, y_test)
    print(f"Accuracy:      {metrics_aps['accuracy']:.2%}")
    print(f"Set Size (SS): {metrics_aps['set_size']:.2f}")
    print(f"Coverage (CR): {metrics_aps['coverage_rate']:.2%}")

    # Exemples détaillés
    print("\n" + "=" * 60)
    print("EXEMPLES DÉTAILLÉS")
    print("=" * 60)

    labels_str = ['A', 'B', 'C', 'D']
    for idx in range(min(3, len(probs_test))):
        example = dataset[n_cal + idx]
        print(f"\nExemple #{idx + 1}:")
        print(f"  Q: {example['question'][:80]}...")
        print(f"  Probas: {dict(zip(labels_str, probs_test[idx].round(3)))}")
        print(f"  Vraie réponse: {labels_str[y_test[idx]]}")

        C_lac = cp_lac.predict(probs_test[idx])
        C_aps = cp_aps.predict(probs_test[idx])

        print(f"  C(X) LAC: {[labels_str[i] for i in C_lac]}")
        print(f"  C(X) APS: {[labels_str[i] for i in C_aps]}")

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"{'Méthode':<10} {'Accuracy':<12} {'Set Size':<12} {'Coverage':<12}")
    print("-" * 46)
    print(f"{'LAC':<10} {metrics_lac['accuracy']:<12.2%} {metrics_lac['set_size']:<12.2f} {metrics_lac['coverage_rate']:<12.2%}")
    print(f"{'APS':<10} {metrics_aps['accuracy']:<12.2%} {metrics_aps['set_size']:<12.2f} {metrics_aps['coverage_rate']:<12.2%}")


if __name__ == "__main__":
    main()