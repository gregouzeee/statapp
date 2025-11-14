# run_batched_selfcheck.py
import os
import argparse
import logging
from dotenv import load_dotenv
import numpy as np
from LLM_judge_gemini import SelfCheckGeminiBatch

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
    parser.add_argument("--model", default="models/gemini-2.5-flash-lite-preview-06-17")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout (seconds)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    

    # === Exemple =======
    sentences = [
        "Albert Einstein was born in Germany.",
        "Albert Einstein was born in France.",
        "Albert Einstein developed the theory of relativity.",
    ]
    passages = [
        "Albert Einstein was born in Ulm, Germany, in 1879. He is best known for developing the theory of relativity.",
        "Einstein, a German-born physicist born in 1879, developed relativity theory.",
        "Born in Germany in 1879, Einstein would become one of the greatest physicists.",
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
