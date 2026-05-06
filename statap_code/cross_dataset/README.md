# `cross_dataset/` — Liens transversaux entre les sections du rapport

Ce dossier contient les scripts ajoutés pour faire dialoguer les
différentes sections du rapport final (`rapport_new.pdf`) :

* **§3** — TriviaQA-numérique : confiance ↔ proximité à la vérité
* **§4** — GSM8K Chain-of-Thought : log-probs vs erreurs
* **§5** — TriviaQA / WEPR : détection d'hallucinations
* (datasets non utilisés dans le rapport actuel : ELI5, MMLU)

Trois questions transverses sont traitées :

1. **Idée 5** — appliquer l'**entropie sémantique** (Kuhn 2023, déjà
   utilisée sur ELI5) au TriviaQA-numérique de §3.
2. **Idée 2** — entraîner un modèle **WEPR sur GSM8K** (avec long CoT)
   et **transférer** WEPR-TriviaQA → GSM8K et inversement.
3. **Idée 1** — produire une **table croisée AUC/ECE/Brier** unifiée
   (scores × datasets) pour synthétiser §3, §4, §5 et les datasets
   ELI5/MMLU.

---

## Logs en temps réel

Tous les scripts de génération (`se_triviaqa_numeric.py`,
`generate_gsm8k_topk.py`) écrivent :

* une **barre de progression `tqdm`** sur `stderr` (mininterval 2 s)
* un **checkpoint texte** sur `stdout` toutes les **10 questions**
  avec : nombre traité, % d'avancement, ok/err, accuracy courante,
  débit (q/s), ETA, et nombre de rate-limits 429.

Pour suivre l'avancée en temps réel sur un cloud / VM :

```bash
# Lancer en arrière-plan en gardant les deux flux séparés :
python se_triviaqa_numeric.py --num 500 --concurrency 8 \
    > logs/se_progress.log 2> logs/se_tqdm.log &

# Suivre le checkpoint texte :
tail -f logs/se_progress.log

# (et la barre tqdm dans logs/se_tqdm.log)
```

Si tu lances en interactif, les deux flux s'affichent ensemble dans
le terminal. Les checkpoints sont **flushés immédiatement**, donc
pas besoin de `python -u` ou `stdbuf`.

Les agrégateurs (`judge_gsm8k.py`, `train_wepr_gsm8k.py`,
`transfer_wepr.py`, `cross_table.py`) sont tous rapides
(quelques secondes à 1 minute pour `cross_table` qui parse les 73k
lignes WEPR) et logguent leur progression toutes les 5 000 lignes.

---

## Pré-requis

* Variables d'environnement (`.env` à la racine du projet) :
  * `GCP_PROJECT`, `GCP_LOCATION` — accès Vertex AI Gemini.
* Python : `requirements.txt` du projet (gemini, scikit-learn, scipy, tqdm,
  numpy, python-dotenv).
* Données externes déjà présentes :
  * `statap_code/distance_verite/triviaqa_numeric_results.jsonl` (§3)
  * `statap_code/WEPR/data/triviaqa_judged.jsonl` (§5)
  * `statap_code/WEPR/data/wepr_model.json`     (modèle WEPR §5)
  * `statap_code/text_uncertainty/eli5_uncertainty/eli5_*.jsonl`
  * `statap_code/Conformal_prediction/mmlu_500_confidence_probs.jsonl` (§2)
  * `statap_code/logit_gsm8k/dataset_gsm8k.jsonl` (questions GSM8K nues)

---

## Ordre d'exécution

### Bloc 1 — Entropie sémantique sur TriviaQA-numérique (Idée 5)

```bash
cd statap_code/cross_dataset
python se_triviaqa_numeric.py --num 500 --concurrency 8
```

Génère `data/se_triviaqa_numeric.jsonl`. Pour chaque question :
K = 5 réponses à T = 1.0, clustering numérique exact (fast path) ou
NLI Gemini 2.5-Flash (fallback), puis `semantic_entropy =
−Σ P(C) log P(C)` sur les clusters pondérés par les log-probs moyens.

Sortie typique : ~5–7 s/question avec 8 workers, soit ~30 min pour 500.
Coût API : ~2 500 appels génération + ~quelques NLI quand les clusters
ne sont pas tranchés numériquement.

### Bloc 2 — WEPR sur GSM8K (Idée 2)

#### 2a) Génération avec top-K log-probs

```bash
python generate_gsm8k_topk.py --num_questions 300 --concurrency 6 \
    --logprobs_k 10 --max_output_tokens 1024 --temperature 1.0
```

Sortie : `data/gsm8k_topk.jsonl`. Chaque ligne contient
`token_data` (toute la génération), `reasoning_token_data`,
`answer_token_data`, et `answer_value` parsé.

#### 2b) Annotation correct/incorrect (pas d'appel LLM, gold dispo)

```bash
python judge_gsm8k.py
```

Sortie : `data/gsm8k_judged.jsonl` avec un champ booléen
`judge_correct` (compatible `WEPR/train_wepr.py`).

#### 2c) Entraînement WEPR sur GSM8K (3 sous-séquences)

```bash
python train_wepr_gsm8k.py --K 10 --n_splits 5
```

Sortie : `data/wepr_gsm8k_model.json`. Contient les coefficients
β/γ et l'AUC en 5-folds CV pour chaque sous-séquence (full /
reasoning / answer-only) — comparable au tableau 4 du rapport.

#### 2d) Transfert TriviaQA ↔ GSM8K

```bash
python transfer_wepr.py --gsm8k_field token_data
```

Sortie : `data/transfer_results.json`. Table 2×2 d'AUC :

|                    | testé sur TriviaQA | testé sur GSM8K |
|--------------------|-------------------|-----------------|
| modèle TriviaQA    | (in-domain)       | (transfert)     |
| modèle GSM8K       | (transfert)       | (in-domain)     |

Le rapport de §4 annonce 0.729 AUC sur GSM8K avec un modèle à
64 features ; ce script permet de comparer à WEPR (2K = 20 features).

### Bloc 3 — Combler les trous de la table croisée

Les fichiers d'origine `triviaqa_numeric_results.jsonl` (§3) et
`eli5_judged.jsonl` ne stockent pas les distributions top-K par token
— seulement les agrégats. EPR et WEPR n'y sont donc pas calculables.
Et `cross_table.py` n'avait pas SE pour GSM8K.

Trois scripts régénèrent ces signaux manquants :

```bash
# (3a) Régénérer TriviaQA-numérique avec logprobs_k=10 — ~10 min
python regenerate_triviaqa_num_topk.py --concurrency 8

# (3b) Régénérer ELI5 avec logprobs_k=10 — ~5 min
python regenerate_eli5_topk.py --concurrency 4

# (3c) Entropie sémantique sur GSM8K (K=5 generations × 1319 q) — ~30 min
python se_gsm8k.py --concurrency 4
```

Outputs respectifs : `data/triviaqa_num_topk.jsonl`, `data/eli5_topk.jsonl`,
`data/se_gsm8k.jsonl`. Toutes joinables aux fichiers existants par
`question` (TriviaQA-num) ou `question_id` (ELI5, GSM8K).

### Bloc 4 — Table croisée (Idée 1)

À lancer **après** les blocs 1, 2 et 3 (sinon les colonnes EPR/WEPR
sur TriviaQA-num et ELI5 seront vides). Le script tourne quand même
si certains fichiers manquent — il met juste plus de cellules `--`.

```bash
python cross_table.py
```

Sorties dans `data/` :

* `cross_table.csv`       — long format (1 ligne par (dataset, score))
* `cross_table_wide.csv`  — pivot AUC, score × dataset
* `cross_table.tex`       — deux tableaux LaTeX (AUC + ECE) prêts à
  inclure dans le rapport via `\input{...}`.

Datasets pris en compte :

* TriviaQA-num (3 397) — §3
* TriviaQA-WEPR (73 793) — §5
* ELI5 (100) — texte libre, judge sur 5
* GSM8K (300, généré par 2a) — §4 régénéré
* MMLU (500) — §2

Scores pris en compte (selon disponibilité par dataset) :
verbalisé, judge, prob_joint, prob_geo_mean, perplexité, p_min,
EPR, WEPR (modèle pré-entraîné §5), entropie sémantique,
SelfCheckGPT, max_softmax (MMLU).

---

## Notes méthodologiques

* **Orientation des scores** : pour l'AUC on oriente automatiquement
  les scores où "haut = mauvais" (perplexité, EPR, SE, SelfCheck) en
  les passant en négatif. Le mapping est dans `HIGHER_IS_BETTER` dans
  `cross_table.py`.

* **ECE / Brier** : calculés uniquement sur les scores interprétables
  comme une probabilité dans [0, 1] (verbalisé, judge, prob_joint,
  prob_geo_mean, p_min, WEPR, max_softmax). Pour les autres
  (perplexité, EPR, SE) on rapporte juste l'AUC.

* **ELI5** : on binarise `judge_scores.accuracy ∈ {1..5}` en
  `correct = (accuracy ≥ 4)`. Choix discutable, à ajuster selon ce
  qu'on veut dire dans le rapport.

* **GSM8K** : la régénération ici (300 q × 1 024 max tokens,
  T = 1.0) ne reproduit pas exactement les 64-features du §4. Mais
  on a un cadre comparable (CoT long sur GSM8K) où WEPR — un modèle
  à 21 paramètres — peut être confronté à une approche riche.

* **TriviaQA-num × TriviaQA-WEPR** : ce sont deux datasets différents
  (3 397 questions numériques vs 73 793 questions générales). Un
  même score (ex. perplexité) peut donc avoir une AUC sensiblement
  différente d'une colonne à l'autre — c'est précisément le genre
  d'écart que la table cherche à révéler.
