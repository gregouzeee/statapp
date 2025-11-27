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

## 1.2 Dataset FinQA 

*(Description à remplir)*

---

### Format 
*(à remplir)*

---

###  Exemple  
*(à remplir)*

---

## 1.3 Dataset FEVER — *Structure prête, à compléter*

*(Description à remplir)*

---

### Format 
*(à remplir)*

---

###  Exemple  
*(à remplir)*

---

# 2. Approches méthodologiques  
*(section à rédiger)*

---

# 3. Modèles utilisés  
*(section à rédiger)*