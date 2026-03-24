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


def compute_counts_by_bin(data, conf_key="verbal_confidence",
                          error_key="abs_error", bin_width=0.1):
    # Bins entre 0 et 1
    bins = np.arange(0, 1 + bin_width, bin_width)
    if bins[-1] < 1.0:
        bins = np.append(bins, 1.0)

    total = np.zeros(len(bins) - 1, dtype=int)
    good = np.zeros(len(bins) - 1, dtype=int)

    for row in data:
        try:
            conf = float(row.get(conf_key, None))
            err = float(row.get(error_key, None))
        except (TypeError, ValueError):
            continue

        # clamp conf in [0,1]
        conf = max(0.0, min(1.0, conf))

        idx = np.searchsorted(bins, conf, side="right") - 1
        if 0 <= idx < len(total):
            total[idx] += 1
            if err == 0.0:
                good[idx] += 1

    bad = total - good
    centers = (bins[:-1] + bins[1:]) / 2.0
    return bins, centers, total, good, bad


def print_bin_summary(bins, total, good, bad):
    print("Bin range\t\tN\tGood\tBad\t%Good")
    for i in range(len(total)):
        left = bins[i]
        right = bins[i + 1]
        n = total[i]
        g = good[i]
        b = bad[i]
        pct = (100.0 * g / n) if n > 0 else 0.0
        print(f"[{left:.2f}, {right:.2f})\t{n}\t{g}\t{b}\t{pct:5.1f}%")


def plot_stacked_histogram(centers, total, good, bad, bin_width, conf_key):
    plt.figure(figsize=(9, 5))

    # Barres empilées: bad (orange) en bas, good (bleu) au-dessus
    # (ou l’inverse si tu préfères ; ici bleu = bonnes réponses)
    plt.bar(centers, bad, width=bin_width * 0.9, label="Bad", color="orange")
    plt.bar(centers, good, width=bin_width * 0.9, bottom=bad, label="Good", color="tab:blue")

    # Annotations: N et %good
    for x, n, g, b in zip(centers, total, good, bad):
        if n <= 0:
            continue
        pct = 100.0 * g / n
        # texte au-dessus de la barre totale
        plt.text(x, b + g + max(0.02 * max(total), 0.5), f"N={n}\n{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.xlabel(f"Confiance ({conf_key})")
    plt.ylabel("Nombre de réponses")
    plt.xlim(0, 1)
    plt.grid(alpha=0.25, axis="y")
    plt.title("Histogramme par tranche de confiance (bleu=bonnes, orange=mauvaises)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--conf_key", default="verbal_confidence")
    parser.add_argument("--error_key", default="abs_error")
    parser.add_argument("--bin_width", type=float, default=0.1)
    args = parser.parse_args()

    data = load_jsonl(args.jsonl_path)

    bins, centers, total, good, bad = compute_counts_by_bin(
        data,
        conf_key=args.conf_key,
        error_key=args.error_key,
        bin_width=args.bin_width
    )

    print_bin_summary(bins, total, good, bad)

    plot_stacked_histogram(
        centers, total, good, bad, args.bin_width, args.conf_key
    )


if __name__ == "__main__":
    main()