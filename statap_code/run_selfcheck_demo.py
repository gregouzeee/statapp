import json
import torch
from pathlib import Path

from statap_code.selfcheckgpt_prompt import SelfCheckLLMPrompt


def main(dataset_path: str = None, model_name: str = "gpt2"):
    if dataset_path is None:
        dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "sample_selfcheck.jsonl"
    else:
        dataset_path = Path(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        data = json.loads(line)

    sentences = data.get("sentences", [])
    passages = data.get("passages", [])

    device = torch.device("cpu")
    model = SelfCheckLLMPrompt(model=model_name, device=device)

    print("Running SelfCheck on dataset:", dataset_path)
    scores = model.predict(sentences, passages, verbose=True)

    for i, s in enumerate(sentences, 1):
        print(f"{i}. {s}\n    → score={scores[i-1]:.4f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
