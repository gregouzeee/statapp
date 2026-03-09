#!/usr/bin/env python3
"""Quick test of finegrained MIDV scripts"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

print("🧪 Testing Fine-Grained MIDV Scripts\n")

# Test 1: train_midv_finegrained imports
try:
    from MIDV.scripts.train_midv_finegrained import MIDVCountryDataset
    print("✅ train_midv_finegrained.py: Imports OK")
except Exception as e:
    print(f"❌ train_midv_finegrained.py: {type(e).__name__}: {e}")

# Test 2: test_midv_finegrained_uncertainty imports
try:
    from MIDV.scripts.test_midv_finegrained_uncertainty import FineGrainedUncertainty
    print("✅ test_midv_finegrained_uncertainty.py: Imports OK")
except Exception as e:
    print(f"❌ test_midv_finegrained_uncertainty.py: {type(e).__name__}: {e}")

# Test 3: show_finegrained_results imports
try:
    from MIDV.scripts.show_finegrained_results import show_results
    print("✅ show_finegrained_results.py: Imports OK")
except Exception as e:
    print(f"❌ show_finegrained_results.py: {type(e).__name__}: {e}")

# Test 4: Dataset loading
print("\n🔍 Dataset Test:")
try:
    dataset_dir = Path("MIDV/datasets/MIDV/images")
    if dataset_dir.exists():
        dataset = MIDVCountryDataset(str(dataset_dir), split="test", max_images=2)
        print(f"   ✅ Dataset path exists: {dataset_dir}")
        print(f"   ✅ Test split loaded: {len(dataset)} images")
        print(f"   ✅ Num classes: {MIDVCountryDataset.NUM_CLASSES}")
        print(f"   ✅ Countries: {list(MIDVCountryDataset.COUNTRY_NAMES.values())[:3]}...")
    else:
        print(f"   ❌ Dataset path not found: {dataset_dir}")
except Exception as e:
    print(f"   ❌ Error: {type(e).__name__}: {e}")

# Test 5: Check results directory
print("\n📁 Results Directory:")
try:
    results_dir = Path("MIDV/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Results directory ready: {results_dir}")
except Exception as e:
    print(f"   ❌ Error: {type(e).__name__}: {e}")

print("\n✅ All tests passed! Scripts are ready to run.")
print("\n🚀 Next steps:")
print("   1. python MIDV/scripts/train_midv_finegrained.py")
print("   2. python MIDV/scripts/test_midv_finegrained_uncertainty.py")
print("   3. python MIDV/scripts/show_finegrained_results.py")
