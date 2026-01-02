import os
import argparse
import logging
from dotenv import load_dotenv
from typing import List, Optional, Tuple

from .white_box import UnifiedProbGeminiBatch

def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def pretty_print_results(queries: List[str], results: List[Optional[Tuple[Optional[object], Optional[float]]]]):
    print("\n=== Résultats (valeur, prob) ===")
    for i, (q, res) in enumerate(zip(queries, results), start=1):
        if res is None:
            print(f"{i:02d}. {q}\n    → (None, None)")
            continue
        val, prob = res
        val_str = str(val) if val is not None else "None"
        prob_str = f"{prob:.6f}" if isinstance(prob, float) else "None"
        print(f"{i:02d}. {q}\n    → ({val_str}, {prob_str})")

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run batched Gemini unified probs (bool, float, or string).")
    parser.add_argument("--model", default="models/gemini-2.5-flash-lite")
    parser.add_argument("--mode", choices=["bool", "float", "string"], required=True, help="Type de sortie attendu.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)


    # ============ Exemples ============
    queries_bool = [
        "Is 7 * 8 equal to 56?",
        "Does 2^10 equal 1000?",
        "Is 13 a prime number?",
    ]

    queries_float = [
        "Compute 17 / 4.",        
        "What is 10 / 3?. approximate to 4 decimal places.",        
        "Evaluate 1e-2 + 0.005.",    
    ]

    queries_str = [
        'Select the prime number. A) 21  B) 25  C) 20  D)23',
        'Select the chemical symbol of water. A) CO2  B) O2  C) H2O  D) N2',
        'Select the largest planet in our solar system. A) Earth  B) Mars  C) Jupiter  D) Venus',
    ]
    # ==================================


    if args.mode == "bool":
        queries = queries_bool
    elif args.mode == "float":
        queries = queries_float
    else:
        queries = queries_str

    checker = UnifiedProbGeminiBatch(
        model=args.model,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0,
        max_output_tokens=64,
    )

    logging.info("Starting predict_probs on %d queries (mode=%s, batch=%d)...",
                 len(queries), args.mode, args.batch_size)

    results = checker.predict_probs(
        queries,
        mode=args.mode,
        batch_size=args.batch_size,
        verbose=True
    )

    logging.info("Done.")
    pretty_print_results(queries, results)

if __name__ == "__main__":
    main()
