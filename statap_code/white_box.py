import re
import time
import json
import math
import logging
from typing import List, Optional, Tuple, Literal
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

Mode = Literal["bool", "float", "string"]

# ---------- Regex de parsing TEXT (fallback/validation JSON) ----------
BOOL_TAIL = re.compile(r"^\s*(true|false)\s*(?=[,\]\}])", re.IGNORECASE)
FLOAT_TAIL = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?=[,\]\}])")
UPPER_LETTER_RE = re.compile(r'[A-Z]')

class UnifiedProbGeminiBatch:
    """
    Batch de queries → JSON strict:
      - mode="bool"   : {"answers":[true,false,...]}
      - mode="float"  : {"answers":[0,-5,3.14,1e-3,...]}
      - mode="string" : {"answers":["A","C","D",...]}

    Retourne [(value, prob)] aligné sur les queries:
      value : bool|float|str|None selon mode
      prob  : float|None (produit des probas des sous-tokens générés pour la valeur)
    """

    def __init__(
        self,
        model: str = "models/gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 64,
        default_choices: Optional[List[str]] = None
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.default_choices = default_choices or ["A", "B", "C", "D"]
        logger.info(
            "Initialized UnifiedProbGeminiBatch(model=%s, temp=%.2f, max_out=%d)",
            model, temperature, max_output_tokens
        )

    # ------------------ PROMPTS distincts ------------------

    def _build_prompt_bool(self, queries: List[str]) -> str:
        """Construit un prompt JSON strict pour des réponses booléennes."""
        lines = [
            "<ROLE> You are a strict JSON generator. </ROLE>",
            "<Instructions>",
            "You will receive a list of QUERIES.",
            'Return exactly one JSON object: {"answers":[ true|false, ... ]}',
            "The array MUST match QUERIES in length and order.",
            "Use unquoted JSON booleans true/false. No prose, no markdown.",
            "</Instructions>",
            "",
            "QUERIES:",
        ]
        lines += [f"{i+1}. {q}" for i, q in enumerate(queries)]
        lines += ["", 'Return only: {"answers":[...]}']
        return "\n".join(lines)

    def _build_prompt_float(self, queries: List[str]) -> str:
        """Construit un prompt JSON strict pour des réponses numériques (float)."""
        lines = [
            "<ROLE> You are a strict JSON generator. </ROLE>",
            "<Instructions>",
            "You will receive a list of arithmetic/numeric QUERIES.",
            'Return exactly one JSON object: {"answers":[ <number>, <number>, ... ]}',
            "The array MUST match QUERIES in length and order.",
            "Use unquoted JSON numbers (e.g., 0, -5, 3.14, 1e-3). No prose, no markdown.",
            "</Instructions>",
            "",
            "QUERIES:",
        ]
        lines += [f"{i+1}. {q}" for i, q in enumerate(queries)]
        lines += ["", 'Return only: {"answers":[...]}']
        return "\n".join(lines)

    def _build_prompt_string(self, queries: List[str], choices: List[str]) -> str:
        """Construit un prompt JSON strict pour des réponses string QCM."""
        allowed = ", ".join(f'"{c}"' for c in choices)
        lines = [
            "<ROLE> You are a strict JSON generator. </ROLE>",
            "<Instructions>",
            "You will receive a list of multiple-choice QUERIES.",
            f'Each answer MUST be one of: {allowed}.',
            'Return exactly one JSON object: {"answers":[ "<CHOICE>", ... ]}',
            "The array MUST match QUERIES in length and order.",
            "Return only the JSON object. No prose, no markdown.",
            "</Instructions>",
            "",
            "QUERIES:",
        ]
        lines += [f"{i+1}. {q}" for i, q in enumerate(queries)]
        lines += ["", 'Return only: {"answers":[...]}']
        return "\n".join(lines)
    
    


    def _build_prompt(self, queries: List[str], mode: Mode, choices: Optional[List[str]]) -> str:
        """Construit le bon prompt en fonction du mode."""
        if mode == "bool":
            return self._build_prompt_bool(queries)
        elif mode == "float":
            return self._build_prompt_float(queries)
        else:
            return self._build_prompt_string(queries, choices or self.default_choices)



    # ------------------ PARSING TEXTE (validation/fallback) ------------------
    def _anchor_to_answers_array(self, tokens) -> int:
        """Retourne l'index du 1er token APRÈS '[' dans '{"answers":[ ... ]}'.
        Si non trouvé, renvoie len(tokens) (=> tableau vide)."""
        i, N = 0, len(tokens)
        while i < N and "{" not in tokens[i]["token"]:
            i += 1
        if i >= N: 
            return N
        while i < N and "answers" not in tokens[i]["token"]:
            i += 1
        if i >= N: 
            return N
        while i < N and "[" not in tokens[i]["token"]:
            i += 1
        if i >= N: 
            return N
        return i + 1  # après '['

    def _has_next_value_in_same_token(self, s: str, mode: Mode) -> bool:
        """Vrai si la virgule et le début de la valeur suivante sont dans le même token."""
        pos = s.find(',')
        if pos == -1:
            return False
        tail = s[pos+1:]
        if mode == "string":
            return '"' in tail
        if mode == "bool":
            t = tail.strip().lower()
            return t.startswith("true") or t.startswith("false")
        # float
        t = tail.strip()
        return bool(t) and (t[0].isdigit() or t[0] in "+-.")

    def _parse_answers(self, raw_text: str, expected_len: int, mode: Mode,
                       choices: Optional[List[str]]) -> Optional[List[Optional[object]]]:
        
        """Parse le JSON texte retourné par le modèle pour récupérer answers[] typés."""
        if not raw_text:
            return None
        cleaned = re.sub(r"^(```json|```|''')\s*", "", raw_text.strip(),
                         flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"(```|''')\s*$", "", cleaned.strip(), flags=re.MULTILINE)
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            arr = data.get("answers", None)
            if not isinstance(arr, list):
                return None
            out: List[Optional[object]] = []
            allowed = set((choices or self.default_choices)) if mode == "string" else None
            for x in arr[:expected_len]:
                if mode == "bool":
                    if isinstance(x, bool):
                        out.append(x)
                    elif isinstance(x, str):
                        xl = x.strip().lower()
                        out.append(True if xl in ("true", "yes") else False if xl in ("false", "no") else None)
                    else:
                        out.append(None)
                elif mode == "float":
                    if isinstance(x, (int, float)):
                        out.append(float(x))
                    elif isinstance(x, str):
                        xs = x.strip()
                        if FLOAT_TAIL.match(xs + ","):
                            try:
                                out.append(float(xs))
                            except Exception:
                                out.append(None)
                        else:
                            out.append(None)
                    else:
                        out.append(None)
                else:  # string (QCM)
                    if isinstance(x, str):
                        xv = x.strip().upper()
                        if allowed is None or xv in allowed:
                            out.append(xv)
                        else:
                            out.append(None)
                    else:
                        out.append(None)
            while len(out) < expected_len:
                out.append(None)
            return out
        except Exception:
            return None

    # ------------------ EXTRACTION LOGPROBS (nouveau SDK) ------------------

    @staticmethod
    def _extract_chosen_tokens(resp):
        """
         Retourne la séquence *linéaire* des tokens réellement générés,
        sous forme [{'token': str, 'logprob': float}, ...] depuis chosen_candidates.
        """
        out = []
        for cand in getattr(resp, "candidates", []) or []:
            lpr = getattr(cand, "logprobs_result", None)
            if not lpr:
                continue
            for cc in getattr(lpr, "chosen_candidates", []) or []:
                out.append({"token": cc.token, "logprob": float(cc.log_probability)})       
        return out

    @staticmethod
    def _has_next_value_in_same_token(s: str, mode: Mode) -> bool:
        comma = s.find(',')
        if comma == -1:
            return False
        tail = s[comma+1:]
        if mode == "string":
            return '"' in tail
        elif mode == "bool":
            t = tail.strip().lower()
            return t.startswith("true") or t.startswith("false")
        else:  # float
            t = tail.strip()
            return bool(t) and (t[0].isdigit() or t[0] in "+-.")
        
    @staticmethod
    def _drop_until_first_left_brace(tokens):
        """Supprime tout avant le premier '{' pour ignorer les fences ```json etc."""
        for i, t in enumerate(tokens):
            if "{" in t["token"]:
                return tokens[i:]
        return tokens

    def _read_bool_item(self, tokens, i: int):
        """Lit un élément bool à partir de i. Retourne (value, prob, next_i). 
        - value ∈ {True, False, None}
        - prob = exp(sum logprobs) des tokens qui ont contribué aux lettres de 'true'/'false'
        - next_i ne consomme pas la virgule/']' si elle est dans le même token.
        """
        N = len(tokens)
        acc = []
        logps = []
        started = False
        saw_in_token_delim = False
        j = i
        while j < N:
            s = tokens[j]["token"]
            k = 0
            contributed = False
            while k < len(s):
                ch = s[k]
                if not started:
                    if ch.isalpha():
                        started = True
                        acc.append(ch.lower())
                        contributed = True
                    elif ch in [',', ']']:
                        saw_in_token_delim = True
                        break
                else:
                    if ch.isalpha():
                        acc.append(ch.lower())
                        contributed = True
                    else:
                        if ch in [',', ']']:
                            saw_in_token_delim = True
                        break
                k += 1

            if started and contributed:
                logps.append(tokens[j]["logprob"])

            if saw_in_token_delim:
                break
            j += 1
            if j < N and ("," in tokens[j]["token"] or "]" in tokens[j]["token"]):
                break
        text = "".join(acc)
        if text == "true" or text == "false":
            val = (text == "true")
            prob = math.exp(sum(logps)) if logps else None
            return (val, prob, i if saw_in_token_delim else j)
        return (None, None, j)

    def _read_float_item(self, tokens, i: int):
        """Lit un float (nombre JSON) et ne somme que les logprobs du contenu numérique."""
        N = len(tokens)
        acc = []
        logps = []
        started = False
        saw_in_token_delim = False
        j = i
        valid = set("0123456789+-eE.")
        while j < N:
            s = tokens[j]["token"]
            k = 0
            contributed = False
            while k < len(s):
                ch = s[k]
                if not started:
                    if ch in valid:
                        started = True
                        acc.append(ch)
                        contributed = True
                    elif ch in [',', ']']:
                        saw_in_token_delim = True
                        break
                else:
                    if ch in valid:
                        acc.append(ch)
                        contributed = True
                    else:
                        if ch in [',', ']']:
                            saw_in_token_delim = True
                        break
                k += 1

            if started and contributed:
                logps.append(tokens[j]["logprob"])

            if saw_in_token_delim:
                break

            j += 1
            if j < N and ("," in tokens[j]["token"] or "]" in tokens[j]["token"]):
                break

        text = "".join(acc).strip()
        try:
            val = float(text)
        except Exception:
            return (None, None, j)

        prob = math.exp(sum(logps)) if logps else None
        return (val, prob, i if saw_in_token_delim else j)
    
    def _read_string_item(self, tokens, i: int):
        """
        Version minimaliste:
        - Cherche un guillemet ouvrant, puis le PREMIER token qui contient une lettre majuscule [A-Z].
        - Renvoie le littéral JSON complet "\"X\"" (avec guillemets) et prob = exp(logprob de CE token de lettre).
        - Ne tente pas de compter les guillemets ni la virgule.
        - Ne consomme pas le token qui contient une virgule/']' collée: on laisse l'appelant gérer.
        """
        N = len(tokens)
        j = i

        # 1) aller jusqu'au guillemet ouvrant
        saw_open = False
        while j < N:
            if '"' in tokens[j]["token"]:
                saw_open = True
                break
            # si on tombe sur un délimiteur avant ouverture, on laisse l'appelant gérer
            if any(d in tokens[j]["token"] for d in [',', ']']):
                return (None, None, i)
            j += 1
        if not saw_open:
            return (None, None, j)

        # 2) chercher la lettre majuscule [A-Z] (première rencontre)
        j_letter = j
        letter = None
        while j_letter < N:
            s = tokens[j_letter]["token"]
            # si on voit un ']' avant d'avoir trouvé la lettre, on arrête (tableau fini)
            if not letter and ']' in s and '"' not in s:
                return (None, None, j_letter)
            m = UPPER_LETTER_RE.search(s)
            if m:
                letter = m.group(0)
                break
            j_letter += 1
        if letter is None:
            return (None, None, j_letter)
        
        # 3) construire la valeur et la probabilité
        lp = tokens[j_letter]["logprob"]
        prob = math.exp(lp) if lp else None
        val = f"\"{letter}\""
        next_i = j_letter + 1
        return (val, prob, next_i)

    def _parse_string_answers(self, tokens, max_items: int):
        """
        Boucle simple: ancrage, puis répétition de _read_string_item.
        On saute juste les espaces, on laisse la virgule/']' au parseur externe (boucle de séparation).
        """
        i, N = 0, len(tokens)
        while i < N and "{" not in tokens[i]["token"]:
            i += 1
        while i < N and "answers" not in tokens[i]["token"]:
            i += 1
        while i < N and "[" not in tokens[i]["token"]:
            i += 1
        results = []
        while i < N and len(results) < max_items:
            while i < N and tokens[i]["token"].strip() == "":
                i += 1
            if i >= N:
                break
            if "]" in tokens[i]["token"]:
                break

            val, prob, i2 = self._read_string_item(tokens, i)
            results.append((val, prob))
            i = i2
            if i < N and tokens[i]["token"].strip() == ",":
                i += 1

        return results
   
    def _parse_bool_answers(self, tokens, max_items: int):
        i = self._anchor_to_answers_array(tokens)
        N = len(tokens)
        results = []
        while i < N and len(results) < max_items:
            while i < N and tokens[i]["token"].strip() == "":
                i += 1
            if i >= N or "]" in tokens[i]["token"]:
                break

            val, prob, i2 = self._read_bool_item(tokens, i)
            results.append((val, prob))
            i = i2

            # gestion de la virgule (sans avaler le début de la valeur suivante si in-token)
            while i < N:
                s = tokens[i]["token"]
                if "]" in s:
                    break
                if "," in s:
                    if self._has_next_value_in_same_token(s, "bool"):
                        break
                    i += 1
                    break
                i += 1
        return results

    def _parse_float_answers(self, tokens, max_items: int):
        i = self._anchor_to_answers_array(tokens)
        N = len(tokens)
        results = []
        while i < N and len(results) < max_items:
            while i < N and tokens[i]["token"].strip() == "":
                i += 1
            if i >= N or "]" in tokens[i]["token"]:
                break

            val, prob, i2 = self._read_float_item(tokens, i)
            results.append((val, prob))
            i = i2
            while i < N:
                s = tokens[i]["token"]
                if "]" in s:
                    break
                if "," in s:
                    if self._has_next_value_in_same_token(s, "float"):
                        break
                    i += 1
                    break
                i += 1
        return results
   


    def _parse_answers_and_probs_from_tokens(self, tokens, mode: Mode, max_items: int):
        """Parse les réponses + probas directement depuis les tokens générés."""
        if mode == "bool":
            return self._parse_bool_answers(tokens, max_items)
        if mode == "float":
            return self._parse_float_answers(tokens, max_items)
        return self._parse_string_answers(tokens, max_items)

    # ------------------ APPEL GEMINI + ASSEMBLAGE ------------------

    def _call_batch(
        self,
        queries_chunk: List[str],
        mode: Mode,
        choices: Optional[List[str]] = None
    ) -> List[Tuple[Optional[object], Optional[float]]]:
        """Appel Gemini sur un batch de queries, retourne la liste des (value, prob)."""
        prompt = self._build_prompt(queries_chunk, mode, choices)
        t0 = time.time()
        try:
            logger.debug("Calling Gemini: model=%s, chunk=%d, mode=%s", self.model, len(queries_chunk), mode)
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    response_logprobs=True,  
                    logprobs=1,            
                )
            )
            raw_text = getattr(resp, "text", "") or ""
            parsed_vals = self._parse_answers(raw_text, expected_len=len(queries_chunk), mode=mode, choices=choices)

            # 1) tokens choisis (avec logprobs)
            tokens = self._extract_chosen_tokens(resp)
            if not tokens:
                logger.warning("No token-level logprobs (chosen_candidates) returned; probabilities will be None.")
                return [(val, None) for val in (parsed_vals or [None] * len(queries_chunk))]

            # 2) ignorer tout avant le premier '{' (fences ```json)
            tokens = self._drop_until_first_left_brace(tokens)

            # 3) parser valeurs + probas directement depuis les tokens
            pairs = self._parse_answers_and_probs_from_tokens(tokens, mode=mode, max_items=len(queries_chunk))

            # 4) assembler (merge avec parse texte pour robustesse sur la valeur)
            results: List[Tuple[Optional[object], Optional[float]]] = []
            for idx in range(len(queries_chunk)):
                val = parsed_vals[idx] if (parsed_vals and idx < len(parsed_vals)) else None
                prob = None
                if idx < len(pairs):
                    v2, p2 = pairs[idx]
                    if v2 is not None:
                        val = v2
                    prob = p2
                results.append((val, prob))

        except Exception as e:
            logger.exception("Gemini API error: %s", e)
            results = [(None, None)] * len(queries_chunk)

        dt = (time.time() - t0) * 1000.0
        logger.info("Batch done in %.1f ms (%d queries, mode=%s)", dt, len(queries_chunk), mode)
        return results

    # ------------------ API PUBLIQUE ------------------

    def predict_probs(
        self,
        queries: List[str],
        mode: Mode,
        batch_size: int = 32,
        verbose: bool = False,
        choices: Optional[List[str]] = None, 
    ) -> List[Tuple[Optional[object], Optional[float]]]:
        """API publique: prédit les (value, prob) pour chaque query en batch."""
        N = len(queries)
        if N == 0:
            return []

        if verbose:
            logger.setLevel(logging.INFO)

        out: List[Tuple[Optional[object], Optional[float]]] = [None] * N 
        chunk_idx = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk_idx += 1
            logger.info("queries %d..%d | chunk %d | mode=%s", start, end - 1, chunk_idx, mode)
            chunk = queries[start:end]
            chunk_res = self._call_batch(chunk, mode=mode, choices=choices)
            out[start:end] = chunk_res
        return out
