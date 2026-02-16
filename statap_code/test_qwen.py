import torch
from statap_code.selfcheckgpt_prompt import SelfCheckLLMPrompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a small model for the example to keep runtime reasonable
checker = SelfCheckLLMPrompt(
    model="gpt2",
    device=device
)

## example sentences and sampled passages
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


print("Running prompt-based predict() (Yes/No mapping)...")
scores = checker.predict(
    sentences=sentences,
    sampled_passages=sampled_passages,
    verbose=False
)
print("predict() scores:", scores)

print("\nRunning token-level predict_perplexity() (avg log-prob per token)...")
pp_scores = checker.predict_perplexity(
    sentences=sentences,
    sampled_passages=sampled_passages,
    return_mode="avg_logprob"
)
print("predict_perplexity() avg log-prob scores:", pp_scores)

print("\nRunning predict_perplexity() with sum_logprob (longer sentences get larger magnitude)...")
pp_sum_scores = checker.predict_perplexity(
    sentences=sentences,
    sampled_passages=sampled_passages,
    return_mode="sum_logprob"
)
print("predict_perplexity() sum log-prob scores:", pp_sum_scores)