from statap_code.selfcheckgpt_prompt import SelfCheckLLMPrompt
import math
import torch

def main():
    # init (petit modèle pour test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checker = SelfCheckLLMPrompt(model="gpt2", device=device)

    prefix = "La planète est "   # contexte avant le mot suivant
    next_word = "bleue"         # mot dont on veut la probabilité

    print(f"Prefix: {prefix!r}")
    print(f"Next word: {next_word!r}\n")

    # prob du premier token du mot suivant
    p_first = checker.next_token_prob(prefix, next_word, aggregate="first", return_log=False)
    logp_first = checker.next_token_prob(prefix, next_word, aggregate="first", return_log=True)
    print("first-token prob:", p_first, "logp:", logp_first)

    # prob par token
    per_token_probs = checker.next_token_prob(prefix, next_word, aggregate="tokens", return_log=False)
    per_token_logps = checker.next_token_prob(prefix, next_word, aggregate="tokens", return_log=True)
    print("per-token probs:", per_token_probs)
    print("per-token log-probs:", per_token_logps)

    # prob de la suite complète et moyenne par token
    logp_sum = checker.next_token_prob(prefix, next_word, aggregate="sum", return_log=True)
    p_sum = checker.next_token_prob(prefix, next_word, aggregate="sum", return_log=False)
    logp_avg = checker.next_token_prob(prefix, next_word, aggregate="avg", return_log=True)
    p_avg = checker.next_token_prob(prefix, next_word, aggregate="avg", return_log=False)
    print("sequence sum logp:", logp_sum, "-> prob:", p_sum)
    print("avg logp per token:", logp_avg, "-> avg-prob:", p_avg)
    print("perplexity (per token):", math.exp(-logp_avg))

if __name__ == '__main__':
    main()
