# Statapp — Quantification de l'incertitude des LLMs

Projet de statistique appliquée explorant différentes méthodes pour **quantifier la confiance d'un LLM dans ses réponses** : log-probabilités (white-box), confiance verbalisée, SelfCheckGPT, conformal prediction, et un classifieur dédié (WEPR : *Wrong-or-correct from Ensemble Probabilities and Reasoning*).

---

## Prérequis

- **Python** : développé avec `3.12.7`
- Installer les dépendances depuis la racine :

  ```bash
  pip install -r requirements.txt
  ```

- Créer un fichier `.env` à la racine avec votre clé API Gemini (voir `env_example.sh`) :

  ```env
  GEMINI_API_KEY="votre_clé"
  ```

---

## Structure du projet

```
statapp/
├── statap_code/                # Tout le code Python du projet
│   ├── whitebox_method/        # Confiance via log-probabilités Gemini
│   ├── LLM_as_judge_methods/   # LLM-as-judge (style SelfCheckGPT)
│   ├── selfchechGPT_methods/   # Implémentation SelfCheckGPT (prompt + demos)
│   ├── Conformal_prediction/   # Conformal prediction sur MMLU
│   ├── MMLU_analysis_500/      # Analyse logit / confiance verbalisée sur MMLU
│   ├── stats_desc/             # Stats descriptives (GSM8K, MMLU)
│   ├── distance_verite/        # Distance à la vérité sur TriviaQA numérique
│   ├── WEPR/                   # Classifieur WEPR (XGBoost / régression)
│   ├── cross_dataset/          # Transfert WEPR entre datasets
│   ├── crawler/                # Crawling Wikipedia (génération de données)
│   └── ...
├── MIDV/                       # Méthodes d'incertitude fine-grain (scripts + docs)
├── Notes/                      # Notes d'étape
├── articles/                   # PDFs de référence
├── datasets/                   # Petits datasets locaux
├── rapport/                    # Rapport final (LaTeX + figures)
├── requirements.txt
└── env_example.sh
```

---

## 1. White-box : confiance via log-probabilités

Mesure la confiance du modèle pour chaque réponse en exploitant les **log-probabilités des tokens** retournées par Gemini.

### Principe

1. On donne au modèle une liste de questions.
2. On force la réponse en JSON strict selon le mode :
   - `bool` → `{"answers":[true,false,true]}`
   - `float` → `{"answers":[4.25,0.3333,0.015]}`
   - `string` (QCM) → `{"answers":["D","C","A"]}`
3. La classe `UnifiedProbGeminiBatch` parse le JSON, récupère les tokens choisis et leurs logprobs, et calcule un score de confiance via `exp(sum(logprobs))`.
4. Pour chaque question, on obtient un couple `(valeur, probabilité)`.

### Fichiers

- [statap_code/whitebox_method/white_box.py](statap_code/whitebox_method/white_box.py) — classe `UnifiedProbGeminiBatch`
- [statap_code/whitebox_method/main_whitebox.py](statap_code/whitebox_method/main_whitebox.py) — script de démonstration

### Lancement

Depuis la racine du projet :

```bash
python statap_code/whitebox_method/main_whitebox.py --mode bool
python statap_code/whitebox_method/main_whitebox.py --mode float
python statap_code/whitebox_method/main_whitebox.py --mode string
```

Arguments disponibles :

```bash
python statap_code/whitebox_method/main_whitebox.py \
  --mode {bool,float,string} \
  --model models/gemini-2.0-flash \
  --batch-size 16 \
  --log-level INFO
```

Exemple de sortie :

```
01. Is 7 * 8 equal to 56?
    → (True, 0.998231)
02. Does 2^10 equal 1000?
    → (False, 0.957221)
```

---

## 2. LLM-as-judge (SelfCheckGPT)

Suit l'esprit de **SelfCheckGPT (2023)** : un LLM joue le rôle de *juge* pour vérifier si une phrase est supportée par un ensemble de passages.

Pour chaque phrase `s_i` et passage `p_j` :
- `1.0` → supporté
- `0.0` → non supporté
- `0.5` → indéterminé (parse KO)

### Fichiers

- [statap_code/LLM_as_judge_methods/LLM_judge_gemini.py](statap_code/LLM_as_judge_methods/LLM_judge_gemini.py) — classe `SelfCheckGeminiBatch`
- [statap_code/LLM_as_judge_methods/main_LLM_judge.py](statap_code/LLM_as_judge_methods/main_LLM_judge.py) — script de démonstration

Méthodes principales :
- `predict_matrix(sentences, passages)` → matrice `M × K`
- `predict_mean(sentences, passages)` → score moyen par phrase

### Lancement

```bash
python statap_code/LLM_as_judge_methods/main_LLM_judge.py
python statap_code/LLM_as_judge_methods/main_LLM_judge.py --batch-size 16 --log-level INFO
```

---

## 3. SelfCheckGPT (prompt-based)

Implémentation du prompt SelfCheckGPT et démos.

- [statap_code/selfchechGPT_methods/selfcheckgpt_prompt.py](statap_code/selfchechGPT_methods/selfcheckgpt_prompt.py)
- [statap_code/selfchechGPT_methods/run_selfcheck_demo.py](statap_code/selfchechGPT_methods/run_selfcheck_demo.py)
- [statap_code/selfchechGPT_methods/example_next_word.py](statap_code/selfchechGPT_methods/example_next_word.py)

---

## 4. Conformal Prediction

Conformal prediction appliquée à MMLU (500 questions, plusieurs températures).

- [statap_code/Conformal_prediction/functions/conformal_prediction.py](statap_code/Conformal_prediction/functions/conformal_prediction.py) — implémentation
- [statap_code/Conformal_prediction/functions/best_alpha.py](statap_code/Conformal_prediction/functions/best_alpha.py) — recherche du meilleur seuil
- [statap_code/Conformal_prediction/data/](statap_code/Conformal_prediction/data/) — logits MMLU 500 (températures 0.0 / 0.5 / 1.0) et probabilités de confiance

---

## 5. Analyse MMLU 500

Analyses logit et confiance verbalisée sur le sous-ensemble MMLU 500.

- [statap_code/MMLU_analysis_500/main_logit_mmlu_500.py](statap_code/MMLU_analysis_500/main_logit_mmlu_500.py)
- [statap_code/MMLU_analysis_500/confiance_verbalise_mmlu.py](statap_code/MMLU_analysis_500/confiance_verbalise_mmlu.py)

---

## 6. Statistiques descriptives

Notebooks et figures décrivant les datasets utilisés.

- [statap_code/stats_desc/stats_desc_GSM8K_jsonl.ipynb](statap_code/stats_desc/stats_desc_GSM8K_jsonl.ipynb)
- [statap_code/stats_desc/stats_desc_MMLU.ipynb](statap_code/stats_desc/stats_desc_MMLU.ipynb)
- Figures dans [statap_code/stats_desc/GSM8K_plots/](statap_code/stats_desc/GSM8K_plots/) et [statap_code/stats_desc/MMLU_plots/](statap_code/stats_desc/MMLU_plots/)

---

## 7. Distance à la vérité (TriviaQA numérique)

Étude de la distance entre la réponse du modèle et la vérité terrain sur des questions à réponse numérique.

- Code : [statap_code/distance_verite/functions/](statap_code/distance_verite/functions/) (`main.py`, `analyze.py`, `judge.py`, `build_pandas.py`, `bins_accuracy.py`, `hist.py`, `compare_all_conf.py`, `latek.py`)
- Données : [statap_code/distance_verite/data/](statap_code/distance_verite/data/)
- Résultats : [statap_code/distance_verite/analysis_results/](statap_code/distance_verite/analysis_results/) et [statap_code/distance_verite/Latek_outputs/](statap_code/distance_verite/Latek_outputs/)

---

## 8. WEPR — classifieur d'incertitude

WEPR (*Wrong-or-correct from Ensemble Probabilities and Reasoning*) entraîne un classifieur (régression logistique ou XGBoost) à prédire si une réponse du LLM est correcte à partir de features dérivées de l'ensemble des réponses générées.

### Pipeline

1. `generate_dataset.py` — génère un dataset annoté (questions + ensemble de réponses + features).
2. `judge_dataset.py` — fait juger les réponses par un LLM pour obtenir le label `correct/incorrect`.
3. Entraînement :
   - `train_wepr.py` — régression logistique
   - `train_wepr_xgb.py` — XGBoost (modèle complet)
   - `train_wepr_xgb_simple.py` — XGBoost simplifié
4. `evaluate.py` / `visualize.py` — métriques et figures (ROC, PR, distributions de scores, SHAP).

### Fichiers principaux

- [statap_code/WEPR/wepr.py](statap_code/WEPR/wepr.py) — features WEPR
- [statap_code/WEPR/train_wepr.py](statap_code/WEPR/train_wepr.py)
- [statap_code/WEPR/train_wepr_xgb.py](statap_code/WEPR/train_wepr_xgb.py)
- [statap_code/WEPR/train_wepr_xgb_simple.py](statap_code/WEPR/train_wepr_xgb_simple.py)
- [statap_code/WEPR/evaluate.py](statap_code/WEPR/evaluate.py)
- [statap_code/WEPR/visualize.py](statap_code/WEPR/visualize.py)
- [statap_code/WEPR/rapport_wepr.tex](statap_code/WEPR/rapport_wepr.tex) / [wepr_rapport.pdf](statap_code/WEPR/wepr_rapport.pdf)

> ⚠️ Le dossier `statap_code/WEPR/data/` est **gitignoré** (gros fichiers de génération). Il faut le reconstruire via `generate_dataset.py` puis `judge_dataset.py`.

---

## 9. Cross-dataset — transfert WEPR

Étude du transfert d'un modèle WEPR entraîné sur un dataset (ex : GSM8K) vers d'autres (TriviaQA numérique, ELI5).

### Scripts

- Génération de top-k logprobs :
  - [statap_code/cross_dataset/generate_gsm8k_topk.py](statap_code/cross_dataset/generate_gsm8k_topk.py)
  - [statap_code/cross_dataset/regenerate_triviaqa_num_topk.py](statap_code/cross_dataset/regenerate_triviaqa_num_topk.py)
  - [statap_code/cross_dataset/regenerate_eli5_topk.py](statap_code/cross_dataset/regenerate_eli5_topk.py)
- Jugement : [statap_code/cross_dataset/judge_gsm8k.py](statap_code/cross_dataset/judge_gsm8k.py)
- Semantic-entropy baselines : [statap_code/cross_dataset/se_gsm8k.py](statap_code/cross_dataset/se_gsm8k.py), [statap_code/cross_dataset/se_triviaqa_numeric.py](statap_code/cross_dataset/se_triviaqa_numeric.py)
- Entraînement / transfert :
  - [statap_code/cross_dataset/train_wepr_gsm8k.py](statap_code/cross_dataset/train_wepr_gsm8k.py)
  - [statap_code/cross_dataset/transfer_wepr.py](statap_code/cross_dataset/transfer_wepr.py)
- Tableaux croisés et heatmap : [statap_code/cross_dataset/cross_table.py](statap_code/cross_dataset/cross_table.py), [statap_code/cross_dataset/heatmap_cross_table.py](statap_code/cross_dataset/heatmap_cross_table.py)
- Pipelines : [run_all.sh](statap_code/cross_dataset/run_all.sh), [run_all_parallel.sh](statap_code/cross_dataset/run_all_parallel.sh)

Voir aussi le [README dédié](statap_code/cross_dataset/README.md) du dossier.

> ⚠️ Les gros fichiers `.jsonl` générés (`gsm8k_topk`, `gsm8k_judged`, `triviaqa_num_topk`, `eli5_topk`, `se_*`) sont **gitignorés** ; ils sont reproductibles via les scripts `generate_*` / `regenerate_*`.

---

## 10. MIDV — méthodes d'incertitude fine-grain

Dossier dédié à des méthodes d'incertitude fine-grain (notamment ensembling et calibration). Voir :

- [MIDV/README_FINEGRAINED.md](MIDV/README_FINEGRAINED.md)
- [MIDV/README_UNCERTAINTY_METHODS.md](MIDV/README_UNCERTAINTY_METHODS.md)
- [MIDV/Fiche_methodologique.md](MIDV/Fiche_methodologique.md)
- Scripts dans [MIDV/scripts/](MIDV/scripts/)

> ⚠️ `MIDV/datasets/` est **gitignoré** (gros fichiers de données).

---

## Fichiers ignorés par Git

Le `.gitignore` exclut notamment :
- `__pycache__/`, `.venv/`, `.DS_Store`
- `*.pt` (modèles PyTorch)
- `MIDV/datasets/`
- `statap_code/WEPR/data/`
- Les gros `.jsonl` de `statap_code/cross_dataset/data/`
- `tableau de bord.xlsx` (fichier local)
