import torch
from selfcheckgpt_prompt import SelfCheckLLMPrompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checker = SelfCheckLLMPrompt(
    model="meta-llama/Llama-2-7b-chat-hf",
    device=device
)

## il faudra construire le code pour échantillonner ?
sentences = [
    "Albert Einstein was born in Germany.",
    "Albert Einstein was born in France.",
    "Albert Einstein developed the theory of relativity.",
]
sampled_passages = [
    "Albert Einstein was born in Ulm, Germany, in 1879. He is best known for developing the theory of relativity.",
    "Einstein, a German-born physicist born in 1879, developed relativity theory.",
    "Born in Germany in 1879, Einstein would become one of the greatest physicists."
]


scores = checker.predict(
    sentences=sentences,
    sampled_passages=sampled_passages,
    verbose=True
)

print(scores)