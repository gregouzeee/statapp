import numpy as np
import torch
import math
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


class SelfCheckLLMPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None,
        device = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        #nécessaire pour GPT 2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.model = AutoModelForCausalLM.from_pretrained(model, dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        #self.prompt_template = "{context}\n{sentence}\nAnswer only by Yes or No. "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=10,  #was 5
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        # print(text)
        # words=[]
        # word=''
        # for i in range(len(text)):
        #     if text[i] == ' ':
        #         words.append(word)
        #         word=''
        #     else:
        #         word += text[i]
        # words.append(word)
        # print(words)
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]

    @torch.no_grad()
    def _token_logprobs(self, prefix: str, text: str):
        """
        Retourne la liste des log-probs (float) des tokens de `text` conditionnés sur `prefix`.
        """
        # tokenize prefix and full sequence
        pref_enc = self.tokenizer(prefix, return_tensors="pt")
        full_enc = self.tokenizer(prefix + text, return_tensors="pt")
        input_ids = full_enc.input_ids.to(self.device)

        outputs = self.model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

        # indices
        pref_len = pref_enc.input_ids.shape[1]
        seq_len = input_ids.shape[1]

        if seq_len <= pref_len:
            return []

        # compute log softmax once
        log_probs = F.log_softmax(logits, dim=-1)  # (1, seq_len, vocab)

        # for each token position i in (pref_len .. seq_len-1), the distribution predicting token at i
        # is located at logits position i-1
        token_logps = []
        for i in range(pref_len, seq_len):
            prev_pos = i - 1
            token_id = input_ids[0, i].item()
            lp = log_probs[0, prev_pos, token_id].item()
            token_logps.append(lp)

        return token_logps

    @torch.no_grad()
    def predict_perplexity(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        batch_size: int = 32,
        verbose: bool = False,
        return_mode: str = "avg_logprob",  # or 'sum_logprob'
    ):
        """
        Calcule pour chaque `sentence` la log-probabilité conditionnelle donnée chaque `sampled_passage`.
        Retourne la moyenne (sur les passages) d'un score par phrase.

        Le score par phrase peut être la moyenne des log-probs des tokens ('avg_logprob')
        ou la somme ('sum_logprob'). Plus haut = modèle trouve la phrase plus probable.
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        for sent_i in range(num_sentences):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # build a simple prefix: we condition on the context and a label indicating start of sentence
                safe_sample = sample.replace("\n", " ")
                prefix = f"Context: {safe_sample}\n\nSentence: "
                token_logps = self._token_logprobs(prefix, sentence)
                if len(token_logps) == 0:
                    score = 0.0
                else:
                    if return_mode == "sum_logprob":
                        score = float(sum(token_logps))
                    else:
                        score = float(sum(token_logps) / len(token_logps))
                scores[sent_i, sample_i] = score

        return scores.mean(axis=-1)

    @torch.no_grad()
    def next_token_prob(
        self,
        prefix: str,
        next_word: str,
        aggregate: str = "first",  # 'first'|'sum'|'avg'|'tokens'
        return_log: bool = False,
    ):
        """
        Retourne la probabilité (ou log-prob si `return_log=True`) du(s) token(s)
        correspondant à `next_word` conditionné(s) sur `prefix`.

        - `aggregate='first'` : prob du premier token de `next_word` (par défaut)
        - `aggregate='avg'`   : moyenne des log-probs par token (retourne exp(avg) si return_log=False)
        - `aggregate='sum'`   : prob de la suite complète (exp(sum) si return_log=False)
        - `aggregate='tokens'`: renvoie la liste des log-probs par token
        """
        token_logps = self._token_logprobs(prefix, next_word)
        if not token_logps:
            return None

        if aggregate == "tokens":
            if return_log:
                return token_logps
            return [float(math.exp(lp)) for lp in token_logps]

        if aggregate == "first":
            lp = token_logps[0]
        elif aggregate == "sum":
            lp = sum(token_logps)
        elif aggregate == "avg":
            lp = sum(token_logps) / len(token_logps)
        else:
            raise ValueError("Unknown aggregate: choose 'first'|'sum'|'avg'|'tokens'")

        if return_log:
            return float(lp)
        return float(math.exp(lp))