import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from statap_code.entropy_uncertainty.entropy_llm_gemini import GeminiMCQEntropy


def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Entropy-based uncertainty using Gemini on MCQ.")
    parser.add_argument("--model", default="models/gemini-2.5-flash-lite")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing (.env).")

    questions = [
        "Which planet is known as the Red Planet?",
        "What is the chemical symbol for water?",
        "Which number is prime?",
    ]
    choices_list = [
        ["Earth", "Mars", "Jupiter", "Venus"],
        ["CO2", "O2", "H2O", "N2"],
        ["21", "25", "20", "23"],
    ]

    runner = GeminiMCQEntropy(model=args.model, api_key=api_key, temperature=0.0, max_output_tokens=1024)
    results = runner.predict(questions, choices_list)

    for i, (q, res) in enumerate(zip(questions, results), start=1):
        probs = res["probs"]
        print(f"{i:02d}. {q}")
        print(f"    pred={res['pred']}  entropy_norm={res['entropy_norm']:.4f}  confidence={res['confidence']:.4f}")
        print(f"    probs={probs}")


if __name__ == "__main__":
    main()
