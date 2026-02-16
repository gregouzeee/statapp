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
from datasets import load_dataset

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_mmlu_subjects():
    """
    Récupère automatiquement la liste des sujets disponibles dans MMLU.
    """
    from datasets import get_dataset_config_names
    subjects = get_dataset_config_names("cais/mmlu")
    # Filtrer "all" si présent car ce n'est pas un sujet individuel
    subjects = [s for s in subjects if s != "all"]
    return subjects


def build_dataset(questions_per_subject: int = 10, output_path: str = None):
    """
    Charge {questions_per_subject} questions par sujet du dataset MMLU via HuggingFace datasets
    et les combine dans un seul dataset.

    Args:
        split: Le split à utiliser ("test", "validation", "dev")
        questions_per_subject: Nombre de questions à prendre par sujet
        output_path: Chemin optionnel pour sauvegarder le dataset en JSON

    Returns:
        Liste de dictionnaires contenant toutes les questions
    """
    subjects = get_mmlu_subjects()
    all_data = []
    answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    global_id = 0

    logger.info(f"Téléchargement de {questions_per_subject} questions par sujet pour {len(subjects)} sujets...")

    for subject in tqdm(subjects, desc="Chargement des sujets"):
        try:
            if subject not in ['all', 'auxiliary_train']:
                ds = load_dataset("cais/mmlu", subject, split='test')

                # Prendre les n premières questions du sujet
                for i, item in enumerate(ds):
                    if i >= questions_per_subject:
                        break

                    options = item['choices']

                    all_data.append({
                        "id": global_id,
                        "subject": subject,
                        "question": item['question'],
                        "options": {
                            "A": options[0],
                            "B": options[1],
                            "C": options[2],
                            "D": options[3]
                        },
                        "true_answer": answer_map[item['answer']]
                    })
                    global_id += 1

        except Exception as e:
            logger.warning(f"Erreur lors du chargement du sujet '{subject}': {e}")
            continue

    logger.info(f"Dataset créé avec {len(all_data)} questions au total")

    # Sauvegarde optionnelle en JSON
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset sauvegardé dans {output_path}")

    return all_data


#build_dataset(questions_per_subject=10, output_path='statap_code/mmlu_subset.json')

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
    batch_size: int = 20
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

#Charger le dataset mmlu_subset.json
with open('statap_code/mmlu_subset.json', 'r', encoding='utf-8') as f:
    mmlu_data = json.load(f)

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY manquante dans le .env")

print(mmlu_data[0])

# Récupérer les confiances verbalisées
items_to_process = mmlu_data[0:40]
confidences = get_verbalized_confidence_batched(
    client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")),
    model_name="gemini-2.0-flash",
    items=items_to_process,
    batch_size=20
)

# Enrichir le dataset avec les confiances verbalisées
results = []
for item, conf in zip(items_to_process, confidences):
    enriched_item = item.copy()
    enriched_item["verbalized_confidence"] = conf
    # Ajouter la réponse prédite (celle avec la plus haute confiance)
    if conf:
        enriched_item["predicted_answer"] = max(conf, key=conf.get)
        enriched_item["max_confidence"] = conf[enriched_item["predicted_answer"]]
    results.append(enriched_item)

# Export du dataset enrichi
output_path = "statap_code/mmlu_verbalized_confidence.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
logger.info(f"Résultats exportés dans {output_path}")

# Afficher quelques statistiques
correct = sum(1 for r in results if r.get("predicted_answer") == r["true_answer"])
logger.info(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")