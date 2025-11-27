import os
import argparse
import logging
from dotenv import load_dotenv

from .LLM_judge_gemini import SelfCheckGeminiBatch

def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run batched Gemini self-check.")
    parser.add_argument("--model", default="models/gemini-2.0-flash")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout (seconds)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    

    # === Exemple =======
    sentences = [
        "The Eiffel Tower was completed in 1889.",
        "The Eiffel Tower is 450 meters tall.",
        "It was originally intended as a temporary structure.",
        "It was designed by Antoni Gaudí."
    ]
    passages = [
        "The Eiffel Tower, designed by Gustave Eiffel for the 1889 Exposition Universelle in Paris, was completed the same year and quickly became a global symbol of France.",
        "When it was built, the Eiffel Tower was initially intended to be dismantled after twenty years. However, its usefulness as a radio transmission tower ensured its preservation.",
        "The Eiffel Tower stands at approximately 324 meters tall, including its antenna. It remained the tallest man-made structure in the world until 1930."
    ]
    #====================


    checker = SelfCheckGeminiBatch(
        model=args.model,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=64,
        request_timeout=args.timeout,
    )

    logging.info("Starting predict_matrix...")
    mat = checker.predict_matrix(sentences, passages, batch_size=args.batch_size, verbose=True)
    logging.info("Matrix shape: %s", mat.shape)

    print("\n=== Score matrix (M x K) ===")
    print(mat)

    scores = mat.mean(axis=1)
    print("\n=== Mean score per sentence ===")
    print(scores)

    for s, sc in zip(sentences, scores):
        print(f"- {s}\n  score={sc:.3f}")

if __name__ == "__main__":
    main()
