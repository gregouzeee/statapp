# main.py
import time
from pathlib import Path

import numpy as np

from statap_code.Conformal_prediction.Previous_version.conformal_prediction_tools import ConformalPredictor, AlphaTuner
from mmlu_api import collect_mmlu_probs, write_output_jsonl, LABELS_STR


def split_three_ways(n_total: int, cal_ratio: float, val_ratio_of_remaining: float, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    n_cal = int(np.round(n_total * cal_ratio))
    remaining = n_total - n_cal
    n_val = int(np.round(remaining * val_ratio_of_remaining))
    n_test = remaining - n_val

    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal:n_cal + n_val]
    test_idx = perm[n_cal + n_val:]
    return cal_idx, val_idx, test_idx


def main():
    # ---------------- config ----------------
    subject = "high_school_mathematics"
    n_samples = 60

    out_jsonl = Path("statap_code") / "Conformal_prediction" / "output.jsonl"

    # gemini
    model = "models/gemma-3-27b-it"

    temperature = 0.0
    batch_size = 15
    batches_per_group = 15
    sleep_between_groups_s = 60

    # splitting + tuning
    random_seed = 42
    split_cal_ratio = 0.5
    val_ratio_of_remaining = 0.5

    alphas = np.linspace(0.01, 0.5, 50)
    coverage_target = 0.95
    loss_lambdas = [1.0, 5.0, 20.0]

    # ---------------- collect probs ----------------
    probs_mat, labels, meta, examples_by_idx = collect_mmlu_probs(
        subject=subject,
        n_samples=n_samples,
        out_jsonl=out_jsonl,
        model=model,
        temperature=temperature,
        batch_size=batch_size,
        batches_per_group=batches_per_group,
        sleep_between_groups_s=sleep_between_groups_s,
        random_seed=random_seed,
    )

    n_total = len(labels)
    cal_idx, val_idx, test_idx = split_three_ways(n_total, split_cal_ratio, val_ratio_of_remaining, random_seed)

    cal_set = set(map(int, cal_idx))
    val_set = set(map(int, val_idx))

    probs_cal, y_cal = probs_mat[cal_idx], labels[cal_idx]
    probs_val, y_val = probs_mat[val_idx], labels[val_idx]

    # ---------------- alpha tuning ----------------
    tuner_aps = AlphaTuner("aps")
    grid_aps = tuner_aps.run_grid(probs_cal, y_cal, probs_val, y_val, alphas, loss_lambdas=loss_lambdas)
    sel_aps = tuner_aps.select_alphas(grid_aps, coverage_target=coverage_target)

    tuner_lac = AlphaTuner("lac")
    grid_lac = tuner_lac.run_grid(probs_cal, y_cal, probs_val, y_val, alphas, loss_lambdas=loss_lambdas)
    sel_lac = tuner_lac.select_alphas(grid_lac, coverage_target=coverage_target)

    out_dir = str(out_jsonl.parent)

    # Une seule figure utile par score
    p1 = tuner_aps.plot_useful_one_figure(grid_aps, sel_aps, save_dir=out_dir, tag="aps", loss_key="loss_lambda_5.0", show=False)
    p2 = tuner_lac.plot_useful_one_figure(grid_lac, sel_lac, save_dir=out_dir, tag="lac", loss_key="loss_lambda_5.0", show=False)

    print("Saved:", p1)
    print("Saved:", p2)

    # Renvoi clair des alphas optimaux pour chaque option
    print("\nAPS optimal alphas:")
    for k, v in tuner_aps.summarize_alphas(sel_aps).items():
        print(f"  {k}: {v}")

    print("\nLAC optimal alphas:")
    for k, v in tuner_lac.summarize_alphas(sel_lac).items():
        print(f"  {k}: {v}")
        
    # choose final alpha rule (ici: min_size_cov, sinon fallback)
    alpha_aps = sel_aps.alpha_min_size_cov or sel_aps.alpha_cov_max or float(alphas[0])
    alpha_lac = sel_lac.alpha_min_size_cov or sel_lac.alpha_cov_max or float(alphas[0])

    # recalibrate on cal+val
    combined_idx = np.concatenate([cal_idx, val_idx]) if len(val_idx) else cal_idx
    probs_comb, y_comb = probs_mat[combined_idx], labels[combined_idx]

    cp_aps = ConformalPredictor("aps", alpha_aps).calibrate(probs_comb, y_comb)
    cp_lac = ConformalPredictor("lac", alpha_lac).calibrate(probs_comb, y_comb)

    # ---------------- fill meta ----------------
    meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["split"] = {
        "split_cal_ratio": float(split_cal_ratio),
        "val_ratio_of_remaining": float(val_ratio_of_remaining),
        "n_total": int(n_total),
        "n_cal": int(len(cal_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(random_seed),
    }
    meta["alpha_tuning"] = {
        "coverage_target": float(coverage_target),
        "loss_lambdas": [float(x) for x in loss_lambdas],
        "aps": {"grid": grid_aps.to_dict(), "selection": sel_aps.to_dict()},
        "lac": {"grid": grid_lac.to_dict(), "selection": sel_lac.to_dict()},
    }
    meta["final"] = {
        "alpha_used": {"aps": float(alpha_aps), "lac": float(alpha_lac)},
        "q_hat": {"aps": float(cp_aps.q_hat), "lac": float(cp_lac.q_hat)},
    }

    # ---------------- enrich examples ----------------
    for i in range(n_total):
        ex = examples_by_idx.get(i)
        if ex is None:
            ex = {"type": "example", "idx": int(i)}
            examples_by_idx[i] = ex

        split = "cal" if i in cal_set else ("val" if i in val_set else "test")
        ex["split"] = split

        p = probs_mat[i]
        y = int(labels[i])
        pred_idx = int(np.argmax(p))

        C_lac = cp_lac.predict(p)
        C_aps = cp_aps.predict(p)

        ex["true"] = LABELS_STR[y]
        ex["pred"] = LABELS_STR[pred_idx]

        ex["lac"] = {
            "set": [LABELS_STR[int(k)] for k in C_lac],
            "covered": bool(y in C_lac),
            "size": int(len(C_lac)),
            "p_true": float(p[y]),
        }
        ex["aps"] = {
            "set": [LABELS_STR[int(k)] for k in C_aps],
            "covered": bool(y in C_aps),
            "size": int(len(C_aps)),
            "p_true": float(p[y]),
        }

    # ---------------- write final jsonl ----------------
    write_output_jsonl(out_jsonl, meta=meta, examples_by_idx=examples_by_idx)
    print("DONE ->", out_jsonl)


if __name__ == "__main__":
    main()
