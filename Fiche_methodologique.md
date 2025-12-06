# Fiche Méthodologique — Statapp  

---

# Sommaire

## 1. Datasets
- 1.1. **GSM8K** 
- 1.2. **FinQA** 
- 1.3. **FEVER** 

## 2. Méthodes incertitudes 

## 3. Modèles retenus 

---

# 1. Méthodes d’incertitude

Cette section présente les jeux de données utilisés pour évaluer les méthodes d’incertitude appliquées aux modèles de langage.  
Chaque dataset répond à un objectif spécifique (raisonnement arithmétique, raisonnement financier, vérification de faits).

---

## 1.1 Dataset GSM8K — Description méthodologique

GSM8K (*Grade School Math 8K*) est un dataset composé de problèmes mathématiques écrits en langage naturel, centrés sur le **raisonnement arithmétique** de niveau primaire/collège.  

---

### Format adopté pour le projet  
Le dataset a été **converti en format `.jsonl`**, avec **une ligne par problème**, pour faciliter les traitements.

Chaque ligne contient deux colonnes : 
- **"question"** qui contient la question du problème au format **string**
-**"reponse"** qui contient la réponse du problème au format **float**

---

###  Exemple complet (format JSONL)

```json
{"question":"Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?","reponse":18.0}
```

```json
{"question":"A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?","reponse":3.0}
```

---

## 1.2 Dataset FinQA — Financial Numerical Question Answering

FinQA (*Financial Numerical Question Answering*) est un dataset composé de questions nécessitant un **raisonnement numérique complexe** sur des données financières, combinant tableaux structurés et texte non-structuré.

---

### Format adopté pour le projet  
Le dataset est utilisé au **format `.json`**, avec **une entrée par question**.

Chaque entrée contient plusieurs champs principaux :
- **"pre_text"** : Texte contextuel avant le tableau financier au format **string**
- **"post_text"** : Texte contextuel après le tableau au format **string**
- **"table"** : Tableau financier structuré au format **liste de listes**
- **"qa"** : Dictionnaire contenant la question et les annotations
  - **"question"** : La question à résoudre au format **string**
  - **"program"** : Programme de raisonnement annoté (séquence d'opérations)
  - **"exe_ans"** : Réponse d'exécution (gold answer) au format **numérique**
  - **"gold_inds"** : Indices des faits supportant la réponse

---

###  Taille du dataset
- **Total** : 8 281 paires question-réponse
- **Training set** : ~6 211 exemples (75%)
- **Dev set** : ~1 159 exemples (14%)
- **Test set** : ~911 exemples (11%)
- **Source** : 2 800 rapports financiers (S&P 500 earnings reports)

---

###  Caractéristiques
- Raisonnement multi-étapes requis (opérations arithmétiques enchaînées)
- Combinaison de données structurées (tableaux) et non-structurées (texte)
- Programmes de raisonnement annotés par des experts financiers
- 62.5% des questions nécessitent uniquement des données tabulaires
- 23.4% nécessitent uniquement du texte
- 14% nécessitent à la fois tableaux et texte

---

###  Exemple complet (format JSON)

```json
{
  "pre_text": ["Management's Discussion and Analysis of Results of Operations"],
  "post_text": ["Operating expenses increased due to higher personnel costs."],
  "table": [
    ["", "2019", "2018"],
    ["Revenue", "5829", "5735"],
    ["Operating expenses", "3214", "3105"]
  ],
  "qa": {
    "question": "What is the change in revenue from 2018 to 2019?",
    "program": ["subtract(", "5829", "5735", ")"],
    "exe_ans": "94",
    "gold_inds": {"table": [0], "text": []}
  }
}
```

---

## 1.3 Dataset FEVER — Fact Extraction and VERification

FEVER (*Fact Extraction and VERification*) est un dataset composé d'affirmations (*claims*) à vérifier à partir d'articles Wikipédia, centré sur la **vérification de faits** (fact-checking).

---

### Format adopté pour le projet  
Le dataset est utilisé au **format `.jsonl`**, avec **une ligne par affirmation**.

Chaque ligne contient trois colonnes principales :
- **"claim"** qui contient l'affirmation à vérifier au format **string**
- **"label"** qui contient l'étiquette de vérité au format **string** (SUPPORTS, REFUTES, NOT ENOUGH INFO)
- **"evidence"** qui contient les passages Wikipédia servant de preuves au format **liste**

---

###  Taille du dataset
- **Training set** : 185 168 affirmations
- **Dev set** : 19 998 affirmations
- **Source** : Articles Wikipédia (version juin 2017)

---

###  Types d'affirmations
- **SUPPORTS** : L'affirmation est vérifiée par les preuves Wikipédia
- **REFUTES** : L'affirmation est contredite par les preuves
- **NOT ENOUGH INFO** : Les preuves disponibles sont insuffisantes pour trancher

---

###  Exemple complet (format JSONL)

```json
{"claim":"The Rodney King riots took place in the most populous county in the USA.","label":"SUPPORTS","evidence":[["Los_Angeles_County,_California",0],["1992_Los_Angeles_riots",0]]}
```

```json
{"claim":"Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.","label":"SUPPORTS","evidence":[["Nikolaj_Coster-Waldau",7],["New_Amsterdam_-LRB-TV_series-RRB-",0]]}
```

```json
{"claim":"Stranger Things is set in Bloomington, Indiana.","label":"REFUTES","evidence":[["Stranger_Things",2]]}
```

---

# 2. Approches méthodologiques  

Cette section présente les différentes méthodes explorées pour quantifier l'incertitude des réponses générées par les Large Language Models (LLMs).

---

## 2.1 SelfCheckGPT — Méthode Black-box

SelfCheckGPT est une méthode permettant d'évaluer l'incertitude d'un LLM en vérifiant la **cohérence de ses réponses** lorsqu'on génère plusieurs réponses pour la même question.

---

### Principe  
- Génération de **5 à 10 réponses** pour une même question (échantillonnage multiple)
- Mesure de la **cohérence** entre ces différentes réponses
- Plus les réponses sont **inconsistantes**, plus le score d'incertitude est élevé

---

### Variantes implémentées  
- **SelfCheckGPT-BERTScore** : Mesure la similarité sémantique entre les phrases générées
- **SelfCheckGPT-NLI** : Utilise un modèle de Natural Language Inference pour détecter les contradictions
- **SelfCheckGPT-Prompt** : Utilise un LLM pour évaluer la cohérence des réponses (LLM-based)

---

### Métrique  
**Score d'inconsistance** entre les échantillons générés  
→ Plus le score est élevé, plus l'incertitude est importante

---

### Caractéristiques  

**Type d'accès** : Black-box (ne nécessite pas d'accès aux logits du modèle)

**Avantages** :  
- Ne nécessite que l'accès aux sorties textuelles du modèle
- Détecte efficacement les hallucinations et incohérences
- Applicable via API à n'importe quel LLM

**Limitations** :  
- Coût computationnel élevé (nécessite de multiples générations)
- Temps d'exécution important pour de grands datasets

---

---

# 3. Modèles utilisés  
*(section à rédiger)*
