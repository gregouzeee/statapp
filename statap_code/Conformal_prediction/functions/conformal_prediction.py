import json
import math
import random
from typing import List, Dict, Any

LABELS = ["A", "B", "C", "D"]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def probs_vector(item: Dict[str, Any]) -> List[float]:
    p = item["probs_abcd"]
    # gère None -> 0.0
    vec = [float(p.get(L, 0.0) or 0.0) for L in LABELS]
    s = sum(vec)
    # si jamais ce n’est pas parfaitement normalisé (rare), renormalise
    if s > 0 and abs(s - 1.0) > 1e-6:
        vec = [v / s for v in vec]
    return vec

def true_label(item: Dict[str, Any]) -> str:
    return item["solution"]["answer_letter"]

def conformal_quantile(scores: List[float], alpha: float) -> float:
    n = len(scores)
    s_sorted = sorted(scores)
    k = math.ceil((n + 1) * (1 - alpha))  # 1-indexed
    k = min(max(k, 1), n)
    return s_sorted[k - 1]

# ---------- LAC ----------
def lac_score(p_vec: List[float], y: str) -> float:
    py = p_vec[LABELS.index(y)]
    return 1.0 - py

def lac_predict_set(p_vec: List[float], qhat: float) -> List[str]:
    thr = 1.0 - qhat
    C = [L for L, pr in zip(LABELS, p_vec) if pr >= thr]
    if not C:
        # fallback: argmax
        C = [LABELS[max(range(4), key=lambda i: p_vec[i])]]
    return C

# ---------- APS ----------
def aps_score(p_vec: List[float], y: str) -> float:
    pairs = sorted(zip(LABELS, p_vec), key=lambda t: t[1], reverse=True)
    py = p_vec[LABELS.index(y)]
    cum = 0.0
    for L, pr in pairs:
        if pr >= py - 1e-15:  # tol pour égalités/ties flottantes
            cum += pr
        else:
            break
    return cum

def aps_predict_set(p_vec: List[float], qhat: float) -> List[str]:
    pairs = sorted(zip(LABELS, p_vec), key=lambda t: t[1], reverse=True)
    C = []
    cum = 0.0
    for L, pr in pairs:
        cum += pr
        if cum <= qhat + 1e-15:
            C.append(L)
        else:
            break
    if not C:
        C = [pairs[0][0]]  # argmax
    return C

# ---------- Evaluation ----------
def evaluate(items: List[Dict[str, Any]], alpha: float, method: str, seed: int = 0, cal_frac: float = 0.5):
    rnd = random.Random(seed)
    items = items[:]  # copy
    rnd.shuffle(items)

    n = len(items)
    n_cal = int(round(n * cal_frac))
    cal = items[:n_cal]
    test = items[n_cal:]

    # calibration
    scores = []
    for it in cal:
        p = probs_vector(it)
        y = true_label(it)
        if method == "lac":
            scores.append(lac_score(p, y))
        elif method == "aps":
            scores.append(aps_score(p, y))
        else:
            raise ValueError("method must be 'lac' or 'aps'")

    qhat = conformal_quantile(scores, alpha)

    # test metrics
    cover = 0
    sizes = []
    top1_acc = 0

    for it in test:
        p = probs_vector(it)
        y = true_label(it)

        pred_top1 = LABELS[max(range(4), key=lambda i: p[i])]
        top1_acc += (pred_top1 == y)

        if method == "lac":
            C = lac_predict_set(p, qhat)
        else:
            C = aps_predict_set(p, qhat)

        sizes.append(len(C))
        cover += (y in C)

    return {
        "method": method,
        "alpha": alpha,
        "n_cal": len(cal),
        "n_test": len(test),
        "qhat": qhat,
        "coverage": cover / len(test),
        "avg_set_size": sum(sizes) / len(sizes),
        "top1_acc": top1_acc / len(test),
    }

if __name__ == "__main__":
    path = "statap_code/Conformal_prediction/logit_mmlu_500_temp0_5.jsonl"
    items = load_jsonl(path)

    for alpha in [0.05, 0.1, 0.2]:
        res_lac = evaluate(items, alpha=alpha, method="lac", seed=0, cal_frac=0.5)
        res_aps = evaluate(items, alpha=alpha, method="aps", seed=0, cal_frac=0.5)
        print(res_lac)
        print(res_aps)
        print("-" * 60)
