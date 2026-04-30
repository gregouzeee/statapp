import json
import re
import numpy as np
import matplotlib.pyplot as plt

PATH = "statap_code/gsm8k/answers_logprob.jsonl"

def get_top1(logprobs):
    return [step[0]["logprob"] for step in logprobs if step]

def extract_number(text):
    """
    Extrait un nombre depuis une chaîne.
    Exemples :
    '42' -> 42
    '$42.0' -> 42.0
    'The answer is 42' -> 42
    """
    if text is None:
        return None

    text = str(text).strip().replace(",", "")
    match = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match[-1])  # on prend le dernier nombre trouvé
    except:
        return None

def answers_match(model_answer, gold_answer, tol=1e-9):
    a = extract_number(model_answer)
    b = extract_number(gold_answer)

    if a is None or b is None:
        return str(model_answer).strip() == str(gold_answer).strip()

    return abs(a - b) < tol

def resample_curve(curve, target_len=100):
    if len(curve) < 2:
        return None
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, curve)

correct_curves = []
wrong_curves = []

with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        lp = obj.get("logprobs")
        if not lp:
            continue

        top1 = get_top1(lp)
        if len(top1) < 5:
            continue

        correct = answers_match(obj.get("model_answer"), obj.get("gold_answer"))

        curve = resample_curve(top1, target_len=100)
        if curve is None:
            continue

        if correct:
            correct_curves.append(curve)
        else:
            wrong_curves.append(curve)

correct_curves = np.array(correct_curves)
wrong_curves = np.array(wrong_curves)

print("Nb correct:", len(correct_curves))
print("Nb wrong:", len(wrong_curves))

def plot_group(curves, label):
    median = np.median(curves, axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)

    x = np.arange(len(median))
    plt.plot(x, median, linewidth=2, label=label)
    plt.fill_between(x, q25, q75, alpha=0.2)

plt.figure(figsize=(10, 6))

if len(correct_curves) > 0:
    plot_group(correct_curves, "Correct")
if len(wrong_curves) > 0:
    plot_group(wrong_curves, "Wrong")

plt.xlabel("Relative token position")
plt.ylabel("Top-1 logprob")
plt.title("Logprob trajectories: correct vs wrong answers")
plt.legend()
plt.grid(alpha=0.3)
plt.show()