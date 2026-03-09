#!/usr/bin/env python3
"""
Display MIDV Classification Results
Primary  : Fine-grained 10-class PHOTO  (hard, real acquisition conditions)
Baseline : Fine-grained 10-class SCAN   (easy, flat scanned documents)
"""

import json
from pathlib import Path

SEP = "=" * 100
SEP2 = "-" * 100

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).parent
MIDV_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = MIDV_ROOT / "results"


def _load_json(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def show_status():
    """Print a quick availability summary of all result files."""
    print("\n📁 AVAILABLE RESULT FILES")
    print(SEP2)

    def _file_status(path: Path) -> str:
        if not path.exists():
            return "❌ missing  "
        if path.suffix == ".json":
            try:
                with open(path) as f:
                    d = json.load(f)
                if str(d.get("_status", "")).startswith("PENDING"):
                    return "⏳ pending  "
            except Exception:
                pass
        return "✅ ready    "

    entries = [
        ("📸 PHOTO training results  [NEW]",     RESULTS_DIR / "midv_finegrained_photo_training_results.json"),
        ("📸 PHOTO uncertainty results  [NEW]",   RESULTS_DIR / "midv_finegrained_photo_uncertainty_results.json"),
        ("📸 PHOTO model (.pt)  [NEW]",           RESULTS_DIR / "midv_finegrained_photo_model.pt"),
        ("🖨️  SCAN training results  [baseline]",  RESULTS_DIR / "midv_finegrained_training_results.json"),
        ("🖨️  SCAN uncertainty results  [baseline]",RESULTS_DIR / "midv_finegrained_uncertainty_results.json"),
    ]
    for label, path in entries:
        print(f"  {_file_status(path)}  {label:46s}  {path.name}")
    print()


def show_finegrained(results: dict, label: str = "FINE-GRAINED"):
    """Pretty-print fine-grained (10-class) results."""
    print(f"\n🎯 {label}")
    print(SEP2)

    metrics = results.get("metrics", {})
    print(f"  Accuracy:                    {metrics.get('accuracy', 0)*100:6.2f}%")
    print(f"  Mean Confidence:             {metrics.get('mean_confidence', 0)*100:6.2f}%")
    print(f"  Min / Max Confidence:        "
          f"{metrics.get('min_confidence', 0)*100:.1f}%  /  "
          f"{metrics.get('max_confidence', 0)*100:.1f}%")
    print(f"  Total Samples:               {metrics.get('total_samples', 0):6d}")
    print(f"  Correct Predictions:         {metrics.get('correct_predictions', 0):6d}")

    print("\n  📊 Per-method scores")
    for name, label in [("conformal", "Conformal margin"),
                         ("entropy",   "Entropy confidence"),
                         ("selfcheck", "Max softmax prob ")]:
        m = results.get("per_method", {}).get(name, {})
        if m:
            print(f"    {label}: mean={m.get('mean',0)*100:5.2f}%  "
                  f"std={m.get('std',0)*100:4.2f}%  "
                  f"[{m.get('min',0)*100:.1f}% – {m.get('max',0)*100:.1f}%]")

    print("\n  🔍 Calibration")
    calib = results.get("calibration", {})
    conf_c = calib.get("conf_correct", 0)
    conf_w = calib.get("conf_wrong",   0)
    gap    = calib.get("calibration_gap", 0)
    print(f"    Avg confidence on CORRECT predictions: {conf_c*100:6.2f}%")
    print(f"    Avg confidence on WRONG   predictions: {conf_w*100:6.2f}%")
    print(f"    Calibration gap:                       {gap*100:6.2f}%  ", end="")
    if gap > 0.25:
        print("→ ✅ WELL-CALIBRATED")
    elif gap > 0.15:
        print("→ ⚠️  REASONABLY CALIBRATED")
    else:
        print("→ ❌ POORLY CALIBRATED")

    pairs = results.get("confusion_pairs", {})
    if pairs:
        print("\n  🚫 Top confusion pairs (model confuses A → predicted B)")
        for i, (pair, count) in enumerate(
                sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
            print(f"    {i}. {pair}: {count}×")

    rej = results.get("rejection_thresholds", [])
    if rej:
        print("\n  📈 Rejection thresholds (accuracy / coverage trade-off)")
        print(f"    {'Threshold':>10}  {'Accuracy':>9}  {'Coverage':>9}")
        for row in rej:
            print(f"    {row.get('threshold',0):>10.2f}  "
                  f"{row.get('accuracy',0)*100:>8.1f}%  "
                  f"{row.get('coverage',0)*100:>8.1f}%")


def show_baseline(scan_unc, _doctype=None):
    """Display scan-based results as comparison baseline."""
    print("\n=== SCAN RESULTS  (flat scanned documents -- baseline) ===")
    print(SEP2)
    print("  WARNING: Flat scans are too visually distinct -> model is trivially confident.")
    print("           Use PHOTO results above for genuine uncertainty measurement.\n")

    def _is_ready(d):
        return bool(d and not str(d.get("_status", "")).startswith("PENDING"))

    if _is_ready(scan_unc):
        m = scan_unc.get("metrics", {})
        acc  = m.get("accuracy") or m.get("cnn_accuracy") or 0
        conf = m.get("mean_confidence") or m.get("mean_ensemble_confidence") or 0
        print(f"  CNN Accuracy:       {acc*100:6.2f}%")
        print(f"  Mean Confidence:    {conf*100:6.2f}%")
        print(f"  Total Samples:      {m.get('total_samples', '')}")
        for method in ["conformal", "entropy", "selfcheck"]:
            pm = scan_unc.get("per_method", {}).get(method, {})
            if pm and pm.get("mean") is not None:
                print(f"  {method.capitalize():12s}:  mean={pm.get('mean',0)*100:5.2f}%  "
                      f"std={pm.get('std',0)*100:4.2f}%")
    else:
        print("  (no scan results yet)")


def show_next_steps(has_photo: bool):
    print("\n🚀 NEXT STEPS")
    print(SEP2)
    if not has_photo:
        print("  Photo-based results not yet generated. Run:")
        print("    1. python MIDV/scripts/train_midv_finegrained.py")
        print("       → Trains ResNet18 on 10 countries using PHOTO images (~300s)")
        print("       → Saves: midv_finegrained_photo_model.pt")
        print("    2. python MIDV/scripts/test_midv_finegrained_uncertainty.py")
        print("       → Tests 3 uncertainty methods on photo dataset")
        print("       → Saves: midv_finegrained_photo_uncertainty_results.json")
        print("    3. python MIDV/scripts/show_finegrained_results.py")
        print("       → Re-run this script to see results")
    else:
        print("  ✅ All photo results available.")
        print("  Optional: python MIDV/scripts/test_midv_cnn_vs_llm.py --gemini")
        print("            → Compare CNN vs Gemini on photo documents")


def show_results():
    print(SEP)
    print("  MIDV RESULTS DASHBOARD -- Photo (hard) vs Scan (baseline)".center(100))
    print(SEP)

    photo_unc   = _load_json(RESULTS_DIR / "midv_finegrained_photo_uncertainty_results.json")
    photo_train = _load_json(RESULTS_DIR / "midv_finegrained_photo_training_results.json")
    scan_unc    = _load_json(RESULTS_DIR / "midv_finegrained_uncertainty_results.json")

    def _is_ready(d):
        return bool(d and not str(d.get("_status", "")).startswith("PENDING"))

    show_status()

    if _is_ready(photo_unc):
        show_finegrained(photo_unc,
                         label="PHOTO RESULTS  (real acquisition conditions -- genuine uncertainty)")
    else:
        print("\nWARNING: Photo uncertainty results not yet available (see NEXT STEPS below).")

    if _is_ready(photo_train):
        m   = photo_train.get("metrics", {})
        cfg = photo_train.get("configuration", {})
        ds  = photo_train.get("dataset_sizes", {})
        print("\n=== PHOTO TRAINING INFO ===")
        print(SEP2)
        print(f"  Test Accuracy:   {(m.get('test_accuracy') or 0)*100:6.2f}%  "
              f"| Train time: {m.get('training_time_seconds') or 0:.0f}s  "
              f"| Epochs: {cfg.get('num_epochs')}  "
              f"| Device: {cfg.get('device')}")
        print(f"  Dataset -- Train: {ds.get('train')}  Val: {ds.get('val')}  Test: {ds.get('test')}")

    show_baseline(scan_unc)
    show_next_steps(has_photo=_is_ready(photo_unc))

    print("\n" + SEP)
    print("  Dashboard complete.".center(100))
    print(SEP)


if __name__ == "__main__":
    show_results()

