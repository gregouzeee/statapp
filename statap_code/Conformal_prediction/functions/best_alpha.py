import json, math, random
import pandas as pd

LABELS = ["A", "B", "C", "D"]

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def probs_vector(item):
    p = item["probs_abcd"]
    vec = [float(p.get(L, 0.0) or 0.0) for L in LABELS]
    s = sum(vec)
    if s > 0 and abs(s - 1.0) > 1e-6:
        vec = [v / s for v in vec]
    return vec

def true_label(item):
    return item["solution"]["answer_letter"]

def conformal_quantile(scores, alpha):
    n = len(scores)
    s_sorted = sorted(scores)
    k = math.ceil((n + 1) * (1 - alpha))  # 1-indexed
    k = min(max(k, 1), n)
    return s_sorted[k - 1]

# ---------- LAC ----------
def lac_score(p_vec, y):
    return 1.0 - p_vec[LABELS.index(y)]

def lac_predict_set(p_vec, qhat):
    thr = 1.0 - qhat
    C = [L for L, pr in zip(LABELS, p_vec) if pr >= thr]
    if not C:
        C = [LABELS[max(range(4), key=lambda i: p_vec[i])]]
    return C

# ---------- APS ----------
def aps_score(p_vec, y):
    pairs = sorted(zip(LABELS, p_vec), key=lambda t: t[1], reverse=True)
    py = p_vec[LABELS.index(y)]
    cum = 0.0
    for L, pr in pairs:
        if pr >= py - 1e-15:
            cum += pr
        else:
            break
    return cum

def aps_predict_set(p_vec, qhat):
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
        C = [pairs[0][0]]
    return C

def eval_one_alpha(items, alpha, method, seed=0, cal_frac=0.5):
    rnd = random.Random(seed)
    items = items[:]
    rnd.shuffle(items)

    n = len(items)
    n_cal = int(round(n * cal_frac))
    cal = items[:n_cal]
    test = items[n_cal:]

    # calibration scores
    scores = []
    for it in cal:
        p = probs_vector(it)
        y = true_label(it)
        if method == "lac":
            scores.append(lac_score(p, y))
        else:
            scores.append(aps_score(p, y))

    qhat = conformal_quantile(scores, alpha)

    # test metrics
    cover = 0
    sizes = []

    for it in test:
        p = probs_vector(it)
        y = true_label(it)
        if method == "lac":
            C = lac_predict_set(p, qhat)
        else:
            C = aps_predict_set(p, qhat)

        sizes.append(len(C))
        cover += (y in C)

    coverage = cover / len(test)
    avg_size = sum(sizes) / len(sizes)
    target = 1 - alpha
    violation = max(0.0, target - coverage)

    return {
        "alpha": alpha,
        "qhat": qhat,
        "coverage": coverage,
        "avg_set_size": avg_size,
        "target": target,
        "violation": violation,
    }

def choose_best_alphas(items, method, alphas, lambdas, seed=0, cal_frac=0.5):
    # compute grid once
    grid = []
    for a in alphas:
        row = eval_one_alpha(items, a, method, seed=seed, cal_frac=cal_frac)
        grid.append(row)
    df = pd.DataFrame(grid)

    # Ajout: miscoverage = 1 - coverage (toujours >= 0)
    df["miscoverage"] = (1.0 - df["coverage"]).clip(lower=0.0)

    rows = []

    # ---------------------------
    # Critère 1: minimiser avg_set_size sous contrainte coverage >= target (= 1-alpha)
    # ---------------------------
    feasible = df[df["coverage"] >= df["target"]].copy()
    if len(feasible) > 0:
        best_c1 = feasible.sort_values(["avg_set_size", "alpha"]).iloc[0]
    else:
        # fallback: minimise la miscoverage puis la taille
        best_c1 = df.sort_values(["miscoverage", "avg_set_size", "alpha"]).iloc[0]

    rows.append({
        "method": method,
        "criterion": "C1_min_size_s.t._coverage>=target",
        "lambda": None,
        "alpha_star": float(best_c1["alpha"]),
        "coverage": float(best_c1["coverage"]),
        "target": float(best_c1["target"]),
        "miscoverage": float(best_c1["miscoverage"]),
        "avg_set_size": float(best_c1["avg_set_size"]),
    })

    # ---------------------------
    # Critère 2: minimiser J_lambda = avg_set_size + lambda * miscoverage
    # ---------------------------
    for lam in lambdas:
        tmp = df.copy()
        tmp["objective"] = tmp["avg_set_size"] + float(lam) * tmp["miscoverage"]
        best = tmp.sort_values(["objective", "alpha"]).iloc[0]
        rows.append({
            "method": method,
            "criterion": "C2_penalized_size+lambda*miscoverage",
            "lambda": float(lam),
            "alpha_star": float(best["alpha"]),
            "coverage": float(best["coverage"]),
            "target": float(best["target"]),
            "miscoverage": float(best["miscoverage"]),
            "avg_set_size": float(best["avg_set_size"]),
        })

    # ---------------------------
    # Critère 3: minimiser avg_set_size sous contrainte coverage >= 0.90 (fixe)
    # ---------------------------
    cov_threshold = 0.90
    feasible_90 = df[df["coverage"] >= cov_threshold].copy()

    if len(feasible_90) > 0:
        best_c3 = feasible_90.sort_values(["avg_set_size", "alpha"]).iloc[0]
    else:
        # fallback: minimise l'écart à 0.90 puis la taille
        best_c3 = df.assign(
            gap=(cov_threshold - df["coverage"]).abs()
        ).sort_values(["gap", "avg_set_size", "alpha"]).iloc[0]

    rows.append({
        "method": method,
        "criterion": "C3_min_size_s.t._coverage>=0.90",
        "lambda": None,
        "alpha_star": float(best_c3["alpha"]),
        "coverage": float(best_c3["coverage"]),
        "target": float(cov_threshold),
        "miscoverage": float(max(0.0, 1.0 - best_c3["coverage"])),
        "avg_set_size": float(best_c3["avg_set_size"]),
    })

    return pd.DataFrame(rows)


# ------------------- RUN -------------------
path = "statap_code/Conformal_prediction/mmlu_500_confidence_probs.jsonl"
items = load_jsonl(path)

alphas = [i/10000 for i in range(1, 3001)]   # 0.01 à 0.30
lambdas = [1,5,10]

res_lac = choose_best_alphas(items, "lac", alphas, lambdas, seed=0, cal_frac=0.5)
res_aps = choose_best_alphas(items, "aps", alphas, lambdas, seed=0, cal_frac=0.5)

table = pd.concat([res_lac, res_aps], ignore_index=True)
print(table.to_string(index=False))

# Optionnel: exporter
table.to_csv("best_alpha_table.csv", index=False)
