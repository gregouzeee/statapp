# Statapp

##  Python Version

Développé avec:

```bash
Python 3.12.7
```
---

## Installer les dépendances

Depuis la racine du projet :

```bash
pip install -r requirements.txt
```

---

## Fichier `.env`

Pour faire marcher les différents codes, il faut créer un fichier `.env` à la racine du projet avec :

```env
GEMINI_API_KEY="votre_clé"
```

Vous pouvez voir un exemple dans le fichier `env_example.sh` 

---

## Évaluer un score de confiance avec les méthodes white-box

L’idée de cette partie du projet est de **mesurer la confiance du modèle** pour chaque réponse en exploitant les **log-probabilités des tokens** retournées par Gemini.

## Fichiers importants

- `white_box.py`  
  Contient la classe principale `UnifiedProbGeminiBatch` qui :
  - construit les prompts pour Gemini,
  - appelle l’API avec `response_logprobs=True`,
  - parse le JSON de sortie,
  - extrait les tokens + logprobs,
  - calcule un score de confiance pour chaque réponse.

- `main_whitebox.py`  
  Script en ligne de commande qui montre comment utiliser `UnifiedProbGeminiBatch` pour :
  - des réponses booléennes (`mode=bool`),
  - des réponses numériques (`mode=float`),
  - des réponses QCM (`mode=string`).

---

### Principe général

1. On donne au modèle une liste de questions (*queries*).
2. On force la réponse dans un format JSON strict :

   - Mode `bool` :

     ```json
     {"answers":[true,false,true]}
     ```

   - Mode `float` :

     ```json
     {"answers":[4.25,0.3333,0.015]}
     ```

   - Mode `string` (QCM) :

     ```json
     {"answers":["D","C","A"]}
     ```

3. `UnifiedProbGeminiBatch` parse ce JSON pour récupérer la **valeur** de chaque réponse.
4. En parallèle, le code récupère les **tokens choisis** et leurs **logprobs**.
5. En agrégeant ces logprobs (en gros `exp(sum(logprobs))`), on obtient un **score de confiance** pour chaque réponse.
6. Le script retourne pour chaque query un couple :

   ```text
   (valeur, probabilité)
   ```

---

## Comment lancer `main_whitebox.py`

### Mode booléen

Questions dont la réponse attendue est `True` ou `False`.

```bash
python main_whitebox.py --mode bool
```

---

### Mode numérique (`float`)

Questions qui doivent renvoyer un **nombre**.

```bash
python main_whitebox.py --mode float
```

---

### Mode QCM (`string`)

Questions à choix multiples où la réponse est une lettre (`A`, `B`, `C`, `D`, …).

```bash
python main_whitebox.py --mode string
```

Il suffit de changer les querries dans le fichier `main_whitebox.py` pour traiter d'autres exemples.
---

## Arguments disponibles

```bash
python main_whitebox.py   --mode {bool,float,string}   --model models/gemini-2.0-flash   --batch-size 16   --log-level INFO
```

---

## Format de sortie

Exemple d’affichage :

```text
01. Is 7 * 8 equal to 56?
    → (True, 0.998231)

02. Does 2^10 equal 1000?
    → (False, 0.957221)
```

---



## Evaluer les phrases générer par un LLM (SelfCheckGPT)

Cette méthode suit l’esprit du papier **SelfCheckGPT (2023)** :  
un LLM joue le rôle de *juge* pour vérifier si une phrase est **supportée** par un *ensemble de passages*.

Pour chaque phrase *s_i* et passage *p_j* :

- `true`  → supporté  
- `false` → non supporté  
- parse KO → score neutre `0.5`

Le modèle doit répondre **uniquement** :

```json
{"answers":[true,false,true,...]}
```

---


## Fichiers importants

- `LLM_judge_gemini.py`  
  Contient la classe `SelfCheckGeminiBatch`.

- `main_LLM_judge.py`  
  Script permettant de lancer une évaluation SelfCheck.

---

## Scores retournés

- `1.0` si supporté  
- `0.0` si non supporté  
- `0.5` si indéterminé  

Renvoie :

- `predict_matrix(sentences, passages)` → matrice **M × K**
- `predict_mean(sentences, passages)` → score moyen par phrase

---

## Exemple réaliste (inspiré SelfCheckGPT)

Nous voulons vérifier si un modèle hallucine dans un résumé généré.

### Phrases (résumé du modèle)

```text
1. The Eiffel Tower was completed in 1889.
2. The Eiffel Tower is 450 meters tall.
3. It was originally intended as a temporary structure.
4. It was designed by Antoni Gaudí.
```

### Passages (sources candidates)

```text
A. The Eiffel Tower, designed by Gustave Eiffel for the 1889 Exposition Universelle 
   in Paris, was completed the same year and quickly became a global symbol of France.

B. When it was built, the Eiffel Tower was initially intended to be dismantled after 
   twenty years. However, its usefulness as a radio transmission tower ensured its preservation.

C. The Eiffel Tower stands at approximately 324 meters tall, including its antenna. 
   It remained the tallest man-made structure in the world until 1930.

```

Ici :

- Phrase 1 → supportée (A)  
- Phrase 2 → **non supportée** car la tour ne fait pas 450m  
- Phrase 3 → supportée (B)  
- Phrase 4 → **hallucination claire** (Gaudí n’a rien à voir)

---

## Lancer l’exemple SelfCheck

Exécuter :

```bash
python main_LLM_judge.py
```

Ou avec paramètres :

```bash
python main_LLM_judge.py --batch-size 16 --log-level INFO
```

---

##  Exemple de sortie attendue

```
=== Score matrix (M x K) ===
[[1. 0. 0.]
 [0. 0. 0.]
 [0. 1. 0.]
 [0. 0. 0.]]

=== Mean score per sentence ===
[0.33   0.   0.33 0.  ]

- The Eiffel Tower was completed in 1889.
  score=0.333

- The Eiffel Tower is 450 meters tall.
  score=0.000

- It was originally intended as a temporary structure.
  score=0.333

- It was designed by Antoni Gaudí.
  score=0.000
```


