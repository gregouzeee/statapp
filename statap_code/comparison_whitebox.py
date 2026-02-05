import torch
import logging
import time
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- 1. La Classe Checker (Moteur GPU) ---
class LocalMCQChecker:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Chargement de {model_id} sur {self.device} (4-bit)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()
        
        # Pré-calcul des IDs cibles (A, B, C, D)
        self.target_tokens = self._get_target_token_ids()

    def _get_target_token_ids(self) -> Dict[str, List[int]]:
        targets = ["A", "B", "C", "D"]
        mapping = {}
        for t in targets:
            ids = []
            ids.append(self.tokenizer.encode(t, add_special_tokens=False)[-1])
            ids.append(self.tokenizer.encode(" " + t, add_special_tokens=False)[-1])
            mapping[t] = list(set(ids))
        return mapping

    def _format_prompt(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "You are a multiple-choice answering machine. Select the correct option (A, B, C or D)."},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not text.strip().endswith("assistant"):
             text += "\nAnswer:" 
        return text

    def predict_probs(self, queries: List[str]) -> List[Tuple[str, float]]:
        results = []
        for query in tqdm(queries, desc="Analyse GPU"):
            prompt = self._format_prompt(query)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            next_token_logits = outputs.logits[0, -1, :]
            all_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
            
            scores = {}
            total_mass = 0.0
            for letter, valid_ids in self.target_tokens.items():
                p = sum(all_probs[tid].item() for tid in valid_ids)
                scores[letter] = p
                total_mass += p
            
            if total_mass > 0:
                normalized_scores = {k: v / total_mass for k, v in scores.items()}
                best_choice = max(normalized_scores, key=normalized_scores.get)
                best_prob = normalized_scores[best_choice]
            else:
                best_choice = "None"
                best_prob = 0.0
            
            results.append((best_choice, best_prob))
        return results

# --- 2. Fonction pour charger MMLU ---
def load_mmlu_data(subject: str, limit: int = 5):
    """
    Charge le dataset, formate les questions et garde la vraie réponse.
    """
    logger.info(f"Téléchargement MMLU: {subject} (limit={limit})...")
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        logger.error(f"Erreur MMLU: {e}")
        return [], []

    formatted_queries = []
    ground_truths = []
    
    # Mapping 0->A, 1->B...
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for i, item in enumerate(ds):
        if i >= limit: break
        
        # Construction propre de la string QCM
        q_text = f"{item['question']}\nOptions:\n"
        q_text += f"A) {item['choices'][0]}\n"
        q_text += f"B) {item['choices'][1]}\n"
        q_text += f"C) {item['choices'][2]}\n"
        q_text += f"D) {item['choices'][3]}"
        
        formatted_queries.append(q_text)
        ground_truths.append(int_to_letter[item['answer']])
        
    return formatted_queries, ground_truths

# --- 3. Main ---
def main():
    # PARAMÈTRES
    SUBJECT = "global_facts" # Ex: 'abstract_algebra', 'high_school_mathematics', 'global_facts'
    LIMIT = 20              # Nombre de questions à tester

    # 1. Chargement des données réelles
    queries, true_answers = load_mmlu_data(SUBJECT, limit=LIMIT)
    
    if not queries:
        return

    # 2. Initialisation Modèle
    try:
        checker = LocalMCQChecker(model_id="Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"Erreur modèle: {e}")
        return

    # 3. Prédiction
    print(f"\n--- Démarrage de l'analyse sur MMLU ({SUBJECT}) ---")
    predictions = checker.predict_probs(queries)

    # 4. Affichage des résultats
    print("\n" + "="*60)
    print(f" RÉSULTATS MMLU : {SUBJECT}")
    print("="*60 + "\n")

    score = 0
    sum_prob = 0
    for i, (query, (pred_choice, pred_prob), truth) in enumerate(zip(queries, predictions, true_answers), 1):
        is_correct = (pred_choice == truth)
        if is_correct: score += 1
        sum_prob += pred_prob
        
        icon = "✅" if is_correct else "❌"
        
        print(f"QUESTION {i}")
        print(query) 
        print("-" * 30)
        print(f"Vraie réponse : {truth}")
        print(f"Choix Modèle  : {pred_choice} (Log prob: {pred_prob:.2%})")
        print(f"Résultat      : {icon}")
        print("\n")

    print(f"SCORE FINAL : {score}/{len(queries)} ({score/len(queries):.1%})")
    print(f"CONFIANCE MOYENNE : {sum_prob/len(queries):.1%}")

if __name__ == "__main__":
    main()