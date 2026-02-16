# Vision Uncertainty Quantification for Fraud ID Detection

This module implements a complete pipeline for ID fraud classification with 3 uncertainty quantification methods.

## Structure

```
statap_code/vision_uncertainty/
├── __init__.py                      # Package init
├── entropy_uncertainty.py           # Shannon entropy method
├── perplexity_uncertainty.py        # Log-probability / perplexity method
├── conformal_prediction.py          # Conformal prediction method
├── dataset_loader.py                # EST fraud dataset loader
├── fraud_classifier.py              # Main classifier pipeline
└── test_fraud_classifier.py        # Example usage
```

## Dataset (EST)

The EST dataset contains ~41,800 ID images categorized as:

- **Fraud types** (6 categories, ~5,979 images each):
  - `fraud1_copy_and_move`: Document copying and repositioning
  - `fraud2_face_morphing`: Face morphing techniques
  - `fraud3_face_replacement`: Complete face replacement
  - `fraud4_combined`: Multiple fraud types combined
  - `fraud5_inpaint_and_rewrite`: Inpainting and text rewriting
  - `fraud6_crop_and_replace`: Cropping and replacement

- **Positive class** (~5,979 images):
  - `positive`: Genuine, legitimate IDs

## Uncertainty Methods

### 1. Entropy-Based Uncertainty
Computes Shannon entropy of the softmax distribution over logits.
- **High entropy** = high uncertainty (model confused)
- **Low entropy** = low uncertainty (model confident)
- Formula: $H = -\sum_i p_i \log(p_i)$
- Confidence: `1 - (entropy / max_entropy)`

### 2. Perplexity-Based Uncertainty
Adapted from LLM perplexity to vision: log-probability of predicted class.
- **High log-prob** = model confident (word likely)
- **Low log-prob** = model uncertain (word unlikely)
- Perplexity: $e^{-\log P(\text{class})}$
- Confidence: `sigmoid(log_prob * 5.0)`

### 3. Conformal Prediction
Provides calibrated prediction sets with coverage guarantees.
- Requires calibration on validation set
- Returns prediction set: classes with confidence above threshold
- Efficiency metric: `1 - (set_size / num_classes)`
- Empirical coverage: fraction of true labels in prediction sets

## Usage

### Basic Prediction
```python
from statap_code.vision_uncertainty.fraud_classifier import FraudIDClassifier
from statap_code.vision_uncertainty.dataset_loader import create_dataloaders

# Initialize classifier
classifier = FraudIDClassifier(model_name="resnet50", num_classes=7, device="cuda")

# Create dataloaders
dataloaders = create_dataloaders(root_dir="datasets/EST", batch_size=32)

# Calibrate conformal prediction
classifier.calibrate_conformal(dataloaders['val'])

# Evaluate
metrics = classifier.evaluate(dataloaders['test'], return_uncertainties=True)
print(metrics)

# Predict on batch
result = classifier.predict_batch(images, return_uncertainties=True)
# result contains:
#   - predictions: class indices
#   - pred_probs: softmax probabilities
#   - entropy: Shannon entropy
#   - entropy_confidence: normalized confidence [0, 1]
#   - log_prob: log-probability of predicted class
#   - perplexity: exp(-log_prob)
#   - perplexity_confidence: sigmoid(log_prob * 5)
#   - conformal_set_size: size of prediction set
#   - conformal_efficiency: 1 - (set_size / 7)
```

### Run Example
```bash
cd /Users/darfilalikenan/statapp
PYTHONPATH=. /Users/darfilalikenan/statapp/.venv/bin/python \
    statap_code/vision_uncertainty/test_fraud_classifier.py
```

## Output Interpretation

For each prediction, you get:

| Method | Output | Range | Interpretation |
|--------|--------|-------|-----------------|
| **Entropy** | Entropy value | [0, ln(7)≈1.95] | Lower = more confident |
| | Confidence | [0, 1] | Higher = more confident |
| **Perplexity** | Log-prob | [-∞, 0] | Higher (less negative) = more confident |
| | Perplexity | [1, ∞] | Lower = more confident |
| | Confidence | [0, 1] | Higher = more confident |
| **Conformal** | Set size | [1, 7] | Lower = more efficient prediction |
| | Efficiency | [0, 1] | Higher = more efficient (confident) |

## Pretrained Models

The classifier uses PyTorch pretrained models from `torchvision.models`:
- ResNet50, ResNet101, ResNet152, etc.
- Download happens automatically on first run
- Final layer is adapted to 7 classes

## Requirements

- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `pillow >= 9.0.0`
- `numpy >= 1.24.0`

Install via:
```bash
pip install -r requirements.txt
```

## Future Extensions

- [ ] Fine-tuning on EST dataset
- [ ] Multi-model ensemble (averaging logits)
- [ ] Temperature scaling for better calibration
- [ ] Integration with explanability tools (attention maps, CAM)
- [ ] Batch-norm statistics adaptation (test-time adaptation)
