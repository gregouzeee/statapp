#!/usr/bin/env python3
"""
🎯 MIDV Interactive Project Navigator
Browse all results, scripts, and documentation
"""

import os
from pathlib import Path


def print_header(text):
    print(f"\n{'=' * 90}")
    print(f"  {text.center(88)}")
    print(f"{'=' * 90}\n")


def print_menu():
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║        🎯 MIDV FINE-GRAINED COUNTRY CLASSIFICATION - INTERACTIVE NAVIGATOR    ║
╚════════════════════════════════════════════════════════════════════════════════╝

📊 PROJECT SUMMARY:
   • Dataset: MIDV (100 images/country × 10 countries = 1000 total)
   • Task: Fine-grained country identification (10 classes)
   • Difficulty: HARD (not trivial ID vs Passport)
   • Methods: Conformal Prediction, Entropy-based, SelfCheck
   • Goal: Test REAL uncertainty on challenging task
   
   ⭐ FINE-GRAINED & PRODUCTION-READY

═════════════════════════════════════════════════════════════════════════════════

Choose an option:

  1️⃣  📊 VIEW RESULTS
      ├─ Show fine-grained classification results
      ├─ Show uncertainty calibration analysis
      ├─ Show hardest country pairs
      └─ View JSON results files
      
  2️⃣  🚀 RUN EXPERIMENTS
      ├─ Train fine-grained model (10 classes)
      ├─ Test uncertainty methods
      ├─ Test CNN vs LLM comparison (Gemini)
      └─ Run full pipeline
      
  3️⃣  📖 READ DOCUMENTATION
      ├─ Project Summary
      ├─ File Index & Navigation
      ├─ Uncertainty Methods Explained
      ├─ Fine-grained Classification Guide
      └─ Calibration & Rejection Strategy
      
  4️⃣  📁 EXPLORE FILES
      ├─ List all Python scripts
      ├─ List all markdown files
      ├─ List trained models
      ├─ List results JSON files
      └─ Show file structure
      
  5️⃣  🎓 LEARN MORE
      ├─ Why fine-grained is harder
      ├─ Conformal Prediction explanation
      ├─ Entropy-based methods explanation
      ├─ Calibration & rejection strategy
      └─ Production deployment tips

  0️⃣  EXIT

═════════════════════════════════════════════════════════════════════════════════
""")


def option_1_view_results():
    """Option 1: View Results"""
    import json
    print_header("📊 VIEW RESULTS")

    RESULTS_DIR = Path(__file__).parent.parent / "results"

    def _pending(path):
        if not path.exists(): return True
        try:
            with open(path) as f: d = json.load(f)
            return str(d.get("_status", "")).startswith("PENDING")
        except Exception: return False

    fg_unc  = RESULTS_DIR / "midv_finegrained_uncertainty_results.json"
    fg_train = RESULTS_DIR / "midv_finegrained_training_results.json"
    fg_model = RESULTS_DIR / "midv_finegrained_model.pt"

    def icon(p): return "✅" if p.exists() and not _pending(p) else ("⏳" if p.exists() else "❌")

    print(f"  {icon(fg_train)}  1. Training results       (midv_finegrained_training_results.json)")
    print(f"  {icon(fg_unc)}   2. Uncertainty results    (midv_finegrained_uncertainty_results.json)")
    print(f"  {icon(fg_model)} 3. (Model file)           (midv_finegrained_model.pt)")
    print(f"       4. Full dashboard (all results at once)")
    print()

    choice = input("Which would you like to see? (1-4, or 0 to go back): ").strip()

    if choice == "1":
        if fg_train.exists() and not _pending(fg_train):
            with open(fg_train) as f: r = json.load(f)
            m   = r.get("metrics", {})
            cfg = r.get("configuration", {})
            ds  = r.get("dataset_sizes", {})
            print("\n🎓 TRAINING RESULTS — 10-class fine-grained model")
            print("-" * 80)
            print(f"  Test Accuracy:      {(m.get('test_accuracy') or 0)*100:6.2f}%")
            print(f"  Test Loss:          {m.get('test_loss') or 0:.6f}")
            print(f"  Training Time:      {m.get('training_time_seconds') or 0:.1f}s")
            print(f"  Epochs:             {cfg.get('num_epochs')}")
            print(f"  Batch Size:         {cfg.get('batch_size')}")
            print(f"  Device:             {cfg.get('device')}")
            print(f"  Dataset — Train:    {ds.get('train')}  Val: {ds.get('val')}  Test: {ds.get('test')}")
            classes = r.get("country_classes", {})
            if classes:
                print(f"\n  Classes: {list(classes.values())}")
        else:
            print("\n⏳ Training results not yet generated.")
            print("   Run: python MIDV/scripts/train_midv_finegrained.py")

    elif choice == "2":
        if fg_unc.exists() and not _pending(fg_unc):
            with open(fg_unc) as f: r = json.load(f)
            m   = r.get("metrics", {})
            cal = r.get("calibration", {})
            pm  = r.get("per_method", {})
            print("\n📊 UNCERTAINTY RESULTS — 10-class country identification")
            print("-" * 80)
            print(f"  Accuracy:            {(m.get('accuracy') or 0)*100:6.2f}%")
            print(f"  Mean Confidence:     {(m.get('mean_confidence') or 0)*100:6.2f}%")
            print(f"  Total Samples:       {m.get('total_samples', 0)}")
            print(f"  Correct:             {m.get('correct_predictions', 0)}")
            print()
            for name in ["conformal", "entropy", "selfcheck"]:
                d = pm.get(name, {})
                if d:
                    print(f"  {name.capitalize():12s} → mean={d.get('mean',0)*100:5.2f}%  "
                          f"std={d.get('std',0)*100:4.2f}%")
            gap = cal.get("calibration_gap") or 0
            print(f"\n  Calibration gap:     {gap*100:6.2f}%  "
                  f"({'✅ well-calibrated' if gap > 0.25 else '⚠️  moderate' if gap > 0.15 else '❌ poor'})")
            pairs = r.get("confusion_pairs", {})
            if pairs:
                print("\n  Top confusion pairs:")
                for p, c in sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {p}: {c}×")
        else:
            print("\n⏳ Uncertainty results not yet generated.")
            print("   Run: python MIDV/scripts/test_midv_finegrained_uncertainty.py")

    elif choice == "3":
        if fg_model.exists():
            import os
            size_mb = os.path.getsize(fg_model) / 1e6
            print(f"\n✅ Model file found: {fg_model.name}  ({size_mb:.1f} MB)")
        else:
            print("\n⏳ Model not yet trained.")
            print("   Run: python MIDV/scripts/train_midv_finegrained.py")

    elif choice == "4":
        print("\n⏳ Running full results dashboard...")
        os.system(f"python {Path(__file__).parent / 'show_finegrained_results.py'}")

    input("\nPress Enter to continue...")


def option_2_run_experiments():
    """Option 2: Run Experiments"""
    print_header("🚀 RUN EXPERIMENTS - Fine-Grained Classification")
    
    print("""
    1. Train Fine-Grained Model (10 classes)
       ⏱️  ~300 seconds
       💾 Saves: midv_finegrained_model.pt (43MB)
       📊 Output: Training logs + JSON results
       
    2. Test Uncertainty Methods
       ⏱️  ~30 seconds  
       📊 Tests: Conformal, Entropy, SelfCheck
       📈 Metrics: Calibration, rejection thresholds
       
    3. Test CNN vs LLM (requires GEMINI_API_KEY)
       ⏱️  Variable (API calls)
       🤖 Validates: 10-country predictions with Gemini
       
    4. Run Full Pipeline (Sequential)
       ⏱️  ~350 seconds total
       🎯 Train + Test + Results
       
⚠️  Note: Fine-grained (10-class) is HARD, expect real uncertainty!

Which would you like to run? (1-4, or 0 to go back): """)
    
    choice = input().strip()
    
    if choice == "1":
        print("\n⏳ Starting fine-grained model training...")
        os.system("python MIDV/scripts/train_midv_finegrained.py")
        print("\n✅ Training complete! Run option 2 to test uncertainty.")
    elif choice == "2":
        print("\n⏳ Testing uncertainty methods...")
        os.system("python MIDV/scripts/test_midv_finegrained_uncertainty.py")
    elif choice == "3":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print("\n⏳ Testing CNN vs Gemini...")
            os.system("python MIDV/scripts/test_midv_cnn_vs_llm.py --gemini --max-per-country 5")
        else:
            print("\n⚠️  GEMINI_API_KEY not set. Skipping...")
    elif choice == "4":
        print("\n⏳ Running full pipeline...")
        os.system("python MIDV/scripts/train_midv_finegrained.py && python MIDV/scripts/test_midv_finegrained_uncertainty.py")
    
    input("\nPress Enter to continue...")


def option_3_documentation():
    """Option 3: Read Documentation"""
    print_header("📖 READ DOCUMENTATION")
    
    docs = {
        "1": ("Project Summary", "MIDV_ENSEMBLE_FINAL_SUMMARY.md"),
        "2": ("File Index & Navigation", "FILE_INDEX.md"),
        "3": ("Detailed Usage Guide", "README_UNCERTAINTY_METHODS.md"),
        "4": ("Uncertainty Methods", "MIDV_UNCERTAINTY_PROJECT_SUMMARY.md"),
        "5": ("Country Classification", "MIDV_CLASSIFICATION_README.md"),
    }
    
    print("\nAvailable Documentation:\n")
    for key, (title, path) in docs.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"  {key}. {exists} {title:30} ({path})")
    
    print(f"\n  0. Back to main menu")
    print(f"\nWhich document would you like to read? (0-{len(docs)}): ", end="")
    
    choice = input().strip()
    
    if choice in docs:
        path = docs[choice][1]
        os.system(f"cat {path} | less")
    
    input("\nPress Enter to continue...")


def option_4_explore():
    """Option 4: Explore Files"""
    print_header("📁 EXPLORE FILES")
    
    print("""
    1. List All Python Scripts
    2. List All Markdown Documentation
    3. List Trained Models
    4. List Results JSON Files
    5. Show Directory Structure
    6. Check File Sizes
    
Which would you like to see? (1-6, or 0 to go back): """)
    
    choice = input().strip()
    
    if choice == "1":
        print("\n📄 Python Scripts:\n")
        os.system("ls -1 *.py 2>/dev/null | head -20")
    elif choice == "2":
        print("\n📖 Markdown Files:\n")
        os.system("ls -1 *.md 2>/dev/null")
    elif choice == "3":
        print("\n🤖 Trained Models:\n")
        os.system("ls -lh statap_code/comparison_results/midv_*model*.pt 2>/dev/null")
    elif choice == "4":
        print("\n📊 Results Files:\n")
        os.system("ls -lh statap_code/comparison_results/midv_*results*.json 2>/dev/null")
    elif choice == "5":
        print("\n📂 Directory Structure (statap_code/comparison_results/):\n")
        os.system("tree -L 2 statap_code/comparison_results/ 2>/dev/null || find statap_code/comparison_results -type f")
    elif choice == "6":
        print("\n📏 File Sizes:\n")
        os.system("du -sh statap_code/comparison_results/* 2>/dev/null")
    
    input("\nPress Enter to continue...")


def option_5_learn():
    """Option 5: Learn More"""
    print_header("🎓 LEARN MORE - Fine-Grained Classification & Uncertainty")
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗

📚 WHY FINE-GRAINED IS BETTER FOR UNCERTAINTY TESTING

Binary Classification (ID vs Passport) - ❌ NOT USEFUL
─────────────────────────────────────────────────────
  Problem: TOO EASY
    • Visually distinct (passports much larger, different format)
    • 100% accuracy trivial to achieve
    • Uncertainty is FAKE (always 99.99%)
    • Doesn't test real uncertainty
  
  Result: Can't evaluate LLM uncertainty properly

Fine-Grained Classification (10 countries) - ✅ MUCH BETTER
──────────────────────────────────────────────────────────
  Advantage: GENUINELY HARD
    • Must distinguish similar documents (5 IDs, 5 passports)
    • Finnish ID ↔ Estonian ID: very similar designs
    • Greek Passport ↔ Latvian: both EU versions
    • Realistic accuracy: 70-85%
    • REAL uncertainty: 50-80% confidence
  
  Result: Can properly test LLM uncertainty & calibration!

───────────────────────────────────────────────────────────────────────────────

🎯 THE 10 COUNTRIES (Classes)

IDs (5 classes):
  1. Albanian ID        (Southeast Europe, Cyrillic text)
  2. Spanish ID         (Latin alphabet, EU design)
  3. Estonian ID        (Nordic, similar to Finnish)
  4. Finnish ID         (Nordic, similar to Estonian)
  5. Slovak ID          (Central Europe, EU design)

Passports (5 classes):
  6. Azerbaijani        (Central Asia, unique design)
  7. Greek             (EU, similar to Latvian)
  8. Latvian           (EU, similar to Greek)
  9. Russian Internal  (Cyrillic, unique format)
  10. Serbian          (Balkans, EU design)

───────────────────────────────────────────────────────────────────────────────

📊 UNCERTAINTY METHODS (3 combined in ensemble)

1️⃣ CONFORMAL PREDICTION
   What:    Margin between top-2 classes
   Range:   [0, 1] where 1 = perfect separation
   Formula: margin = P(top1) - P(top2)
   
   For 10-class task:
   • Easy case (100% sure): margin ≈ 0.95
   • Hard case (confused): margin ≈ 0.10
   • Shows which countries are hard to distinguish

───────────────────────────────────────────────────────────────────────────────

2️⃣ ENTROPY-BASED UNCERTAINTY
   What:    Shannon entropy of probability distribution
   Range:   [0, 1] where 1 = zero entropy (certain)
   Formula: confidence = 1 - (H / log(10))
   
   For 10-class task:
   • Certain: all prob on 1 class → entropy ≈ 0
   • Uncertain: spread across many classes → entropy ≈ 1
   • More informative than binary (more gradations)

───────────────────────────────────────────────────────────────────────────────

3️⃣ SELFCHECK-STYLE VALIDATION
   What:    Maximum softmax probability
   Range:   [0, 1] where 1 = 100% predicted probability
   Formula: confidence = max(softmax(logits))
   
   For 10-class task:
   • Certain: max_prob ≈ 0.98
   • Uncertain: max_prob ≈ 0.35 (confused between several classes)

───────────────────────────────────────────────────────────────────────────────

🔄 ENSEMBLE VOTING
   Combined score = (Conformal + Entropy + SelfCheck) / 3
   
   Why ensemble?
   • Reduces individual method biases
   • If all 3 agree → very confident decision
   • If 3 disagree → genuine confusion detected

───────────────────────────────────────────────────────────────────────────────

📈 CALIBRATION & REJECTION

Calibration = Does model confidence match accuracy?

✅ WELL-CALIBRATED:
   When model says 80% → actually correct 80% of time
   Gap between correct and wrong confidence: BIG

❌ POORLY-CALIBRATED:
   When model says 80% → actually correct 40% of time
   Model is OVERCONFIDENT

🚫 REJECTION STRATEGY:
   
   Threshold 50%: Accept all predictions
      Coverage=100%, Accuracy=75%
   
   Threshold 70%: Reject < 70% confidence
      Coverage=65%, Accuracy=90%
   
   Threshold 90%: Only accept high-confidence
      Coverage=20%, Accuracy=98%

═════════════════════════════════════════════════════════════════════════════════

📚 WHAT TO EXPECT

Good results:
  • Accuracy: 75-85% (non-trivial, not perfect)
  • Confidence: 60-75% average (genuine uncertainty!)
  • Calibration gap: 20-30% (good separation)
  • Confused pairs: Nordic IDs & EU Passports

═════════════════════════════════════════════════════════════════════════════════
""")
    
    input("\nPress Enter to continue...")


def main():
    while True:
        print_menu()
        choice = input("Select option (0-5): ").strip()
        
        if choice == "0":
            print("\n👋 Thanks for using MIDV Navigator! Goodbye!\n")
            break
        elif choice == "1":
            option_1_view_results()
        elif choice == "2":
            option_2_run_experiments()
        elif choice == "3":
            option_3_documentation()
        elif choice == "4":
            option_4_explore()
        elif choice == "5":
            option_5_learn()
        else:
            print("\n❌ Invalid option. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!\n")
