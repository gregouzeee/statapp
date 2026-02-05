import os
import json
import logging
import argparse
import pandas as pd
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re
from tqdm import tqdm

# Import de votre module existant
from white_box import UnifiedProbGeminiBatch
from google import genai
from google.genai import types

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def load_mmlu_subset(subset: str = "abstract_algebra", split: str = "test", limit: int = 10):
    """
    Charge un sous-ensemble du dataset MMLU via HuggingFace datasets.
    """
    from datasets import load_dataset
    logger.info(f"Chargement de MMLU ({subset})...")
    ds = load_dataset("cais/mmlu", subset, split=split)
    
    # Conversion en liste de dictionnaires simple
    data = []
    # On limite le nombre d'exemples pour économiser des tokens/temps
    for i, item in enumerate(ds):
        if i >= limit: break
        
        options = item['choices']
        # MMLU donne l'index (0-3), on convertit en A-D
        answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        data.append({
            "id": i,
            "question": item['question'],
            "options": {
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3]
            },
            "true_answer": answer_map[item['answer']]
        })
    return data

def format_query_for_selection(item: Dict) -> str:
    """Prépare la query pour la phase 'LogProbs' (choix de la réponse)."""
    q = item['question']
    opts = item['options']
    return f"{q}\nOptions:\nA) {opts['A']}\nB) {opts['B']}\nC) {opts['C']}\nD) {opts['D']}"

def clean_json_markdown(text: str) -> str:
    """Nettoie les balises markdown ```json ... ```."""
    cleaned = re.sub(r"^(```json|```|''')\s*", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned

def get_verbalized_confidence_batched(
    client: genai.Client, 
    model_name: str, 
    items: List[Dict], 
    batch_size: int = 15
) -> List[Dict[str, float]]:
    """
    Phase 2 (BATCHÉE) : Demande au modèle de verbaliser ses probabilités pour plusieurs questions à la fois.
    """
    logger.info(f"Début de la phase verbalisation (Batch size: {batch_size})...")
    
    all_results = [None] * len(items)
    
    # Prompt Système pour forcer le format liste JSON
    sys_prompt = (
        "You are a calibration assistant. You will receive a list of numbered questions.\n"
        "For EACH question, estimate the probability that options A, B, C, and D are correct.\n"
        "You must return a SINGLE JSON object containing a list 'confidences'.\n"
        "The list must preserve the order of the questions.\n\n"
        "Format example:\n"
        "{\n"
        '  "confidences": [\n'
        '    {"A": 0.1, "B": 0.8, "C": 0.05, "D": 0.05},\n'
        '    {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.4}\n'
        "  ]\n"
        "}\n"
        "Strict JSON only. No prose."
    )

    # Découpage en chunks
    for i in range(0, len(items), batch_size):
        chunk = items[i : min(i + batch_size, len(items))]
        logger.info(f"Verbalisation batch {i}-{i+len(chunk)}...")
        
        # Construction du prompt "multi-questions"
        user_content_lines = ["QUESTIONS:"]
        for idx, item in enumerate(chunk):
            q_text = format_query_for_selection(item)
            # On compacte un peu le texte pour économiser des tokens d'entrée
            user_content_lines.append(f"--- Q{idx+1} ---\n{q_text}")
            
        user_content = "\n".join(user_content_lines)

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[sys_prompt, user_content],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            raw_text = clean_json_markdown(response.text)
            parsed = json.loads(raw_text)
            
            batch_confs = parsed.get("confidences", [])
            
            # Vérification de l'alignement
            if len(batch_confs) != len(chunk):
                logger.warning(f"Mismatch taille batch: reçu {len(batch_confs)}, attendu {len(chunk)}. Padding avec zéros.")
                # Fallback simple
                while len(batch_confs) < len(chunk):
                    batch_confs.append({"A":0, "B":0, "C":0, "D":0})
            
            # Assignation dans le tableau global
            for k, conf in enumerate(batch_confs):
                # Nettoyage des clés/types
                clean_conf = {key: float(conf.get(key, 0.0)) for key in ['A', 'B', 'C', 'D']}
                all_results[i + k] = clean_conf

        except Exception as e:
            logger.error(f"Erreur batch verbalisation: {e}")
            # Remplissage avec des valeurs vides pour ne pas décaler les index
            for k in range(len(chunk)):
                all_results[i + k] = {"A": 0, "B": 0, "C": 0, "D": 0}
            time.sleep(1)

    return all_results

def main():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY manquante dans le .env")

    parser = argparse.ArgumentParser(description="MMLU: LogProbs vs Verbalized Confidence")
    parser.add_argument("--model", default="models/gemma-3-27b-it")
    parser.add_argument("--subject", default="high_school_mathematics", help="Sujet MMLU")
    parser.add_argument("--limit", type=int, default=10, help="Nombre de questions à traiter")
    parser.add_argument("--batch-size", type=int, default=5, help="Taille des batchs pour la verbalisation")
    args = parser.parse_args()

    # 1. Chargement
    items = load_mmlu_subset(args.subject, limit=args.limit)
    logger.info(f"{len(items)} questions chargées.")

    # 2. Phase White Box (LogProbs) - UNE PAR UNE
    queries = [format_query_for_selection(item) for item in items]
    
    wb_checker = UnifiedProbGeminiBatch(
        model=args.model,
        api_key=api_key,
        temperature=0.0,
        max_output_tokens=50 # Petit token output car on fait 1 par 1
    )
    
    logger.info("--- Phase 1: White Box (LogProbs) [Mode Sécurisé 1 par 1] ---")
    
    wb_results = []
    
    for i, query in tqdm(enumerate(queries),desc="WhiteBox LogProbs", total=len(queries)):
        logger.info(f"Traitement LogProb question {i+1}/{len(queries)}...")
        
        success = False
        retry_delay = 10
        
        # Tentative avec réessai simple
        for attempt in range(3):
            try:
                # On envoie une liste contenant UNE seule query
                res = wb_checker.predict_probs([query], mode="string", batch_size=1, verbose=False)
                wb_results.extend(res)
                success = True
                break
            except Exception as e:
                logger.warning(f"Erreur API Q{i+1} (Essai {attempt+1}): {e}")
                time.sleep(retry_delay)
                retry_delay *= 2 # Backoff exponentiel
        
        if not success:
            logger.error(f"Échec définitif pour Q{i+1}. Ajout placeholder.")
            wb_results.append((None, 0.0))
            
        # --- LIMITEUR DE VITESSE ---
        # 15 RPM = 1 req / 4 sec. On met 5 sec pour être safe.
        time.sleep(5) 

    # 3. Phase Verbalisée (Introspection)
    # On garde le batching ici car c'est une seule requête pour X questions
    client = genai.Client(api_key=api_key)
    logger.info("--- Phase 2: Verbalisation (Introspection) ---")
    verb_results = get_verbalized_confidence_batched(
        client, 
        args.model, 
        items, 
        batch_size=args.batch_size
    )

    # 4. Consolidation
    final_data = []
    
    # On itère sur la longueur minimum au cas où il y ait eu un décalage
    safe_len = min(len(items), len(wb_results), len(verb_results))
    
    for i in range(safe_len):
        item = items[i]
        wb_val, wb_prob = wb_results[i]
        verb_dict = verb_results[i]
        
        # Le choix du modèle (A, B, C, D)
        model_choice = wb_val if wb_val else "None"
        
        # Récupération de la probabilité verbalisée pour CE choix
        verbalized_conf_for_choice = verb_dict.get(model_choice, 0.0) if model_choice in verb_dict else None

        # Si le modèle a donné une logprob pour son choix (wb_prob), on calcule l'écart
        wb_prob_val = wb_prob if wb_prob is not None else 0.0
        verb_prob_val = verbalized_conf_for_choice if verbalized_conf_for_choice is not None else 0.0
        
        delta = abs(wb_prob_val - verb_prob_val)

        row = {
            "id": item['id'],
            "question": item['question'][:40] + "...",
            "ground_truth": item['true_answer'],
            "model_choice": model_choice,
            "correct": model_choice == item['true_answer'],
            "logprob": round(wb_prob, 4) if wb_prob else None,
            "verbalized": verbalized_conf_for_choice,
            "delta": round(delta, 4),
            "all_verbalized": str(verb_dict)
        }
        final_data.append(row)

    # 5. Affichage
    df = pd.DataFrame(final_data)
    print("\n=== RÉSULTATS COMPARATIFS (LogProbs vs Verbalized) ===")
    cols = ["id", "ground_truth", "model_choice", "correct", "logprob", "verbalized", "delta"]
    print(df[cols].to_string(index=False))
    
    csv_name = f"mmlu_calibration_{args.subject}.csv"
    df.to_csv(csv_name, index=False)
    logger.info(f"Sauvegardé dans {csv_name}")

if __name__ == "__main__":
    main()