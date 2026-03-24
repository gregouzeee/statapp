import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_xy(data, conf_key="verbal_confidence", dist_key="abs_error"):
    xs = []
    ys = []

    for row in data:
        try:
            conf = float(row.get(conf_key, None))
            dist = float(row.get(dist_key, None))
        except (TypeError, ValueError):
            continue

        if conf is None or dist is None:
            continue

        # clamp confidence to [0,1]
        conf = max(0.0, min(1.0, conf))

        xs.append(dist)
        ys.append(conf)

    if len(xs) < 2:
        return None, None

    return np.array(xs), np.array(ys)


def linear_regression(x, y):
    # y = a*x + b
    a, b = np.polyfit(x, y, 1)

    # prédictions
    y_pred = a * x + b

    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return a, b, r2


def plot_regression(x, y, a, b, conf_key, dist_key):
    plt.figure(figsize=(8, 5))

    plt.scatter(x, y, alpha=0.4, s=25)

    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, linestyle="--")

    plt.xlabel(dist_key)
    plt.ylabel(conf_key)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3)
    plt.title("Linear regression: confidence vs distance")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--conf_key", default="verbal_confidence")
    parser.add_argument("--dist_key", default="abs_error")
    args = parser.parse_args()

    data = load_jsonl(args.jsonl_path)

    x, y = extract_xy(data, args.conf_key, args.dist_key)
    if x is None:
        print("Not enough valid data.")
        return

    a, b, r2 = linear_regression(x, y)

    print("=== Linear Regression ===")
    print(f"Model: confidence = a * distance + b")
    print(f"a (slope)     = {a:.6f}")
    print(f"b (intercept) = {b:.6f}")
    print(f"R^2           = {r2:.6f}")

    plot_regression(x, y, a, b, args.conf_key, args.dist_key)


if __name__ == "__main__":
    main()