"""
Adaptation simplifiée de SelfCheckGPT-Prompt pour GPT2
Basé sur : https://github.com/potsawee/selfcheckgpt
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


class SelfCheckPrompt:
    """
    Version simplifiée de SelfCheckGPT-Prompt
    Compatible avec GPT2 et autres modèles causaux
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        prompt_template: str = None
    ):
        """
        Initialise le modèle pour SelfCheck
        
        Args:
            model_name: Nom du modèle HuggingFace (ex: "gpt2", "gpt2-medium")
            device: "cpu" ou "cuda". Si None, détection automatique
            prompt_template: Template custom (optionnel)
        """
        print(f"🔄 Chargement du modèle {model_name}...")
        
        # Détection automatique du device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Chargement du tokenizer
        print("📝 Chargement du tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Nécessaire pour GPT2 (pas de pad_token par défaut)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Chargement du modèle
        print("🧠 Chargement du modèle...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto"
        )
        self.model.eval()  # Mode évaluation
        self.model.to(self.device)
        
        # Template de prompt
        if prompt_template is None:
            self.prompt_template = (
                "Context: {context}\n\n"
                "Sentence: {sentence}\n\n"
                "Is the sentence supported by the context above? "
                "Answer Yes or No.\n\n"
                "Answer:"
            )
        else:
            self.prompt_template = prompt_template
        
        # Mapping réponse -> score
        self.text_mapping = {
            'yes': 0.0,  # Cohérent
            'no': 1.0,   # Incohérent (hallucination)
            'n/a': 0.5   # Incertain
        }
        
        # Pour tracker les réponses inattendues
        self.unknown_responses = set()
        
        print(f"✅ Modèle {model_name} chargé sur {self.device}")
        print(f"💾 Mémoire GPU utilisée: {self._get_gpu_memory()}")
    
    def _get_gpu_memory(self) -> str:
        """Retourne la mémoire GPU utilisée (si disponible)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            return f"{allocated:.2f} GB"
        return "N/A (CPU)"
    
    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = True,
        max_new_tokens: int = 5,
        batch_responses: bool = False
    ) -> np.ndarray:
        """
        Calcule les scores d'inconsistance pour chaque phrase
        
        Args:
            sentences: Liste des phrases à évaluer
            sampled_passages: Liste des passages échantillonnés
            verbose: Afficher la progression
            max_new_tokens: Nombre max de tokens à générer
            batch_responses: Si True, retourne aussi les réponses brutes
            
        Returns:
            scores: Array numpy de shape (len(sentences),)
                   Valeurs entre 0.0 (cohérent) et 1.0 (incohérent)
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        
        # Matrice pour stocker tous les scores
        scores_matrix = np.zeros((num_sentences, num_samples))
        
        # Optionnel : stocker les réponses brutes
        if batch_responses:
            all_responses = []
        
        # Boucle sur les phrases
        iterator = tqdm(range(num_sentences), disable=not verbose, 
                       desc="Évaluation des phrases")
        
        for sent_idx in iterator:
            sentence = sentences[sent_idx]
            
            if batch_responses:
                sentence_responses = []
            
            # Boucle sur les échantillons
            for sample_idx, sample in enumerate(sampled_passages):
                
                # 1. Construction du prompt
                sample_clean = sample.replace("\n", " ")
                prompt = self.prompt_template.format(
                    context=sample_clean,
                    sentence=sentence
                )
                
                # 2. Tokenisation
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024  # Limite pour éviter les erreurs
                ).to(self.device)
                
                # 3. Génération
                try:
                    output_ids = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Déterministe
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    # 4. Décodage
                    output_text = self.tokenizer.decode(
                        output_ids[0],
                        skip_special_tokens=True
                    )
                    
                    # 5. Extraction de la réponse
                    response = output_text.replace(prompt, "").strip()
                    
                    # 6. Post-traitement et scoring
                    score = self._text_to_score(response)
                    scores_matrix[sent_idx, sample_idx] = score
                    
                    if batch_responses:
                        sentence_responses.append(response)
                
                except Exception as e:
                    print(f"\n⚠️  Erreur pour phrase {sent_idx}, sample {sample_idx}: {e}")
                    scores_matrix[sent_idx, sample_idx] = 0.5  # Score neutre en cas d'erreur
                    if batch_responses:
                        sentence_responses.append("ERROR")
            
            if batch_responses:
                all_responses.append(sentence_responses)
        
        # Moyenne sur tous les échantillons pour chaque phrase
        final_scores = scores_matrix.mean(axis=1)
        
        if batch_responses:
            return final_scores, all_responses
        
        return final_scores
    
    def _text_to_score(self, text: str) -> float:
        """
        Convertit la réponse textuelle en score
        
        Args:
            text: Réponse générée par le modèle
            
        Returns:
            score: 0.0 (yes), 1.0 (no), ou 0.5 (autre)
        """
        text_clean = text.lower().strip()
        
        # Vérifie si commence par "yes"
        if text_clean[:3] == 'yes':
            return self.text_mapping['yes']
        
        # Vérifie si commence par "no"
        elif text_clean[:2] == 'no':
            return self.text_mapping['no']
        
        # Tout le reste
        else:
            # Log les réponses inattendues (une seule fois)
            if text_clean not in self.unknown_responses:
                print(f"\n⚠️  Réponse inattendue: '{text_clean}'")
                self.unknown_responses.add(text_clean)
            return self.text_mapping['n/a']
    
    def evaluate_passage(
        self,
        passage: str,
        sampled_passages: List[str],
        sentence_splitter=None,
        verbose: bool = True
    ) -> dict:
        """
        Évalue un passage complet (découpage en phrases automatique)
        
        Args:
            passage: Texte à évaluer
            sampled_passages: Passages échantillonnés pour comparaison
            sentence_splitter: Fonction de découpage (si None, utilise split simple)
            verbose: Afficher les détails
            
        Returns:
            dict avec 'sentences', 'scores', 'mean_score'
        """
        # Découpage en phrases
        if sentence_splitter is None:
            # Découpage simple (à améliorer avec spacy si besoin)
            sentences = [s.strip() + '.' for s in passage.split('.') if s.strip()]
        else:
            sentences = sentence_splitter(passage)
        
        if verbose:
            print(f"\n📄 Évaluation de {len(sentences)} phrases...")
        
        # Calcul des scores
        scores = self.predict(sentences, sampled_passages, verbose=verbose)
        
        # Résultats
        results = {
            'sentences': sentences,
            'scores': scores.tolist(),
            'mean_score': float(scores.mean()),
            'max_score': float(scores.max()),
            'num_hallucinations': int((scores > 0.5).sum())  # Seuil arbitraire
        }
        
        if verbose:
            print(f"\n📊 Résultats:")
            print(f"   Score moyen: {results['mean_score']:.3f}")
            print(f"   Score max: {results['max_score']:.3f}")
            print(f"   Hallucinations potentielles: {results['num_hallucinations']}")
        
        return results


def test_simple():
    """Test rapide de la classe"""
    print("=" * 60)
    print("TEST SIMPLE DE SELFCHECKPROMPT")
    print("=" * 60)
    
    # Initialisation
    checker = SelfCheckPrompt(model_name="gpt2")
    
    # Données de test
    sentences = [
        "Michael Alan Weiner was born in 1942.",
        "Michael Alan Weiner was born in 1960.",  # Contradiction
    ]
    
    sampled_passages = [
        "Michael Alan Weiner was born in 1942 and is a radio host.",
        "Michael Alan Weiner was born in 1942 in New York.",
        "Michael Weiner, born 1942, hosts The Savage Nation.",
    ]
    
    # Évaluation
    scores = checker.predict(sentences, sampled_passages, verbose=True)
    
    # Affichage
    print("\n" + "=" * 60)
    print("RÉSULTATS:")
    print("=" * 60)
    for i, (sent, score) in enumerate(zip(sentences, scores)):
        status = "✅ COHÉRENT" if score < 0.5 else "❌ INCOHÉRENT"
        print(f"\nPhrase {i+1}: {sent}")
        print(f"Score: {score:.3f} {status}")


if __name__ == "__main__":
    test_simple()