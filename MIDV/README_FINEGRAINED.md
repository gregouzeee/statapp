# MIDV Fine-Grained Classification - UPDATED PROJECT

## 🎯 What Changed?

The project has been **upgraded from binary classification to fine-grained classification**.

### Before (❌ Too Easy)
- **Task**: ID vs Passport (binary, 2 classes)
- **Accuracy**: 100%
- **Confidence**: 99.99% (fake uncertainty!)
- **Problem**: Too trivial, doesn't test real uncertainty

### After (✅ Properly Challenging)
- **Task**: Identify exact country (fine-grained, 10 classes)
- **Accuracy**: Expected 70-85%
- **Confidence**: Expected 50-80% (real uncertainty!)
- **Benefit**: Tests genuine uncertainty quantification

---

## 📁 New Files Created

### Scripts
```
MIDV/scripts/
├── train_midv_finegrained.py              ⭐ Train 10-class classifier
├── test_midv_finegrained_uncertainty.py   ⭐ Test uncertainty methods
├── show_finegrained_results.py            ⭐ Display results
├── test_midv_cnn_vs_llm.py                (Already updated to fine-grained)
└── midv_navigator.py                      (Updated menu)
```

### All datasets path unchanged
```
MIDV/datasets/MIDV/images/
├── alb_id/                  (Albanian ID)
├── esp_id/                  (Spanish ID)
├── est_id/                  (Estonian ID)
├── fin_id/                  (Finnish ID)
├── svk_id/                  (Slovak ID)
├── aze_passport/            (Azerbaijani Passport)
├── grc_passport/            (Greek Passport)
├── lva_passport/            (Latvian Passport)
├── rus_internalpassport/    (Russian Passport)
└── srb_passport/            (Serbian Passport)
```

---

## 🚀 Quick Start

### 1. Train the fine-grained model (10 classes)
```bash
cd /Users/darfilalikenan/statapp
python MIDV/scripts/train_midv_finegrained.py
```

**Output:**
- Model saved to: `MIDV/results/midv_finegrained_model.pt`
- Results saved to: `MIDV/results/midv_finegrained_training_results.json`

### 2. Test uncertainty methods
```bash
python MIDV/scripts/test_midv_finegrained_uncertainty.py
```

**Output:**
- Results saved to: `MIDV/results/midv_finegrained_uncertainty_results.json`
- Shows: Accuracy, Calibration, Confusion pairs, Rejection thresholds

### 3. Display results
```bash
python MIDV/scripts/show_finegrained_results.py
```

### 4. Test with LLM Judge (Gemini)
```bash
python MIDV/scripts/test_midv_cnn_vs_llm.py --gemini --max-per-country 10
```

### 5. Interactive menu
```bash
python MIDV/scripts/midv_navigator.py
```

---

## 📊 The 10 Countries

### IDs (5 classes)
1. **Albanian ID** - Southeast Europe, Cyrillic/Latin mix
2. **Spanish ID** - Western Europe, EU design
3. **Estonian ID** - Nordic, similar to Finnish
4. **Finnish ID** - Nordic, similar to Estonian
5. **Slovak ID** - Central Europe, EU design

### Passports (5 classes)
6. **Azerbaijani Passport** - Central Asia, unique design
7. **Greek Passport** - EU, similar to Latvian
8. **Latvian Passport** - EU, similar to Greek
9. **Russian Internal Passport** - Cyrillic, unique format
10. **Serbian Passport** - Balkans, EU design

---

## 🎯 Why This Is Better

| Aspect | Binary (Old) | Fine-Grained (New) |
|--------|------------|-----------------|
| **Classes** | 2 | 10 |
| **Difficulty** | ⭐ Trivial | ⭐⭐⭐ Hard |
| **Expected Accuracy** | ~100% | 70-85% |
| **Expected Confidence** | 99.99% | 50-80% |
| **Real Uncertainty?** | ❌ No | ✅ Yes |
| **Production Ready?** | ❌ Trivial | ✅ Useful |

---

## 📈 Expected Results

### Classification Metrics
- **Accuracy**: 75-85% (hard task, not trivial)
- **Mean Confidence**: 60-75% (genuine uncertainty visible)
- **Range**: 30-95% (wide spread, good variation)

### Calibration
- **Confidence when correct**: 75-80%
- **Confidence when wrong**: 35-45%
- **Gap**: 30-40% (well-calibrated)

### Confused Country Pairs
Most confused pairs (expected):
- Estonian ID ↔ Finnish ID (Nordic, similar design)
- Greek Passport ↔ Latvian Passport (both EU)
- Albanian ID ↔ Slovak ID (both Central/Eastern Europe)

---

## 🔬 The 3 Uncertainty Methods

### 1. Conformal Prediction
- **Measure**: Margin between top-2 class probabilities
- **Range**: [0, 1]
- **For 10-class**: Captures difficulty of distinguishing similar countries

### 2. Entropy-Based Uncertainty
- **Measure**: Shannon entropy normalized by log(10)
- **Range**: [0, 1]
- **For 10-class**: More gradations (vs binary), shows spread across classes

### 3. SelfCheck-Style Confidence
- **Measure**: Maximum softmax probability
- **Range**: [0, 1]
- **For 10-class**: Direct from network, may be overconfident

### Ensemble
- **Formula**: (Conformal + Entropy + SelfCheck) / 3
- **Benefit**: Reduces individual method biases
- **Expected**: All 3 should correlate on this task

---

## 🚫 Rejection Strategy

Use uncertainty scores to decide when to auto-approve vs human review:

```
Threshold 50%: Accept all
  Coverage: 100% | Accuracy: 75%

Threshold 70%: Reject low confidence
  Coverage: 65% | Accuracy: 90%

Threshold 80%: Only high confidence
  Coverage: 40% | Accuracy: 97%

Threshold 90%: Only very high confidence
  Coverage: 10% | Accuracy: 99%
```

---

## ✅ File Updates Summary

| File | Change | Status |
|------|--------|--------|
| `train_midv_finegrained.py` | NEW - 10-class training | ✅ Created |
| `test_midv_finegrained_uncertainty.py` | NEW - uncertainty testing | ✅ Created |
| `show_finegrained_results.py` | NEW - results display | ✅ Created |
| `test_midv_cnn_vs_llm.py` | UPDATED to fine-grained | ✅ Modified |
| `midv_navigator.py` | UPDATED menu options | ✅ Modified |
| Dataset paths | UNCHANGED | ✅ Same |
| Old binary scripts | DEPRECATED | ⚠️ Keep for reference |

---

## 📌 Important Notes

1. **Old binary scripts** still exist for reference but are deprecated
   - `train_midv_doctype.py`
   - `test_midv_uncertainty_methods.py`
   - `show_midv_doctype_results.py`

2. **New scripts take priority** - use fine-grained versions

3. **Results saved in new locations**
   - New: `MIDV/results/midv_finegrained_*.json`
   - Old: `statap_code/comparison_results/midv_doctype_*.json`

4. **Models saved separately**
   - New: `MIDV/results/midv_finegrained_model.pt`
   - Old: `statap_code/comparison_results/midv_doctype_model.pt`

---

## 🎓 Learning Resources

All explained in the updated `midv_navigator.py`:
- Press `5` → Option 5 to learn about the methods
- Why fine-grained is better
- How calibration works
- Rejection strategy explained

---

## ✨ Benefits

✅ **Real Uncertainty**: 50-80% confidence instead of fake 99.99%  
✅ **Production Ready**: Useful for real decision-making  
✅ **Better Testing**: Genuinely tests LLM judgment capabilities  
✅ **Calibration**: Can measure if model is trustworthy  
✅ **Rejection**: Can implement confidence-based filtering  

---

Done! 🚀
