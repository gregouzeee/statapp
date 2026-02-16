#!/usr/bin/env python3
"""Quick test of the vision uncertainty pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from statap_code.vision_uncertainty.fraud_classifier import FraudIDClassifier
from statap_code.vision_uncertainty.dataset_loader import create_dataloaders
import torch

print('=== Testing FraudIDClassifier ===')

# Initialize classifier (pretrained=False for fast test)
classifier = FraudIDClassifier(model_name='resnet50', num_classes=7, device='cpu', pretrained=False)
print('✓ Classifier initialized')

# Load small batch of data
dataloaders = create_dataloaders(root_dir='datasets/EST', batch_size=8, num_workers=0)
images, labels = next(iter(dataloaders['test']))
print(f'✓ Loaded batch: {images.shape}, labels: {labels.shape}')

# Predict with uncertainties
result = classifier.predict_batch(images[:4], return_uncertainties=True)
print(f'✓ Predictions made: shape {result["predictions"].shape}')
print(f'✓ Entropy: shape {result["entropy"].shape}')
print(f'✓ Log-prob: shape {result["log_prob"].shape}')
print(f'✓ Perplexity: shape {result["perplexity"].shape}')

# Sample output
print('\nSample prediction:')
for i in range(min(2, len(labels))):
    pred = result['predictions'][i]
    true_label = labels[i].item()
    entropy_conf = result['entropy_confidence'][i]
    perp_conf = result['perplexity_confidence'][i]
    perp = result['perplexity'][i]
    
    class_names = [
        "fraud1: copy_move",
        "fraud2: morphing",
        "fraud3: replacement",
        "fraud4: combined",
        "fraud5: inpaint",
        "fraud6: crop",
        "positive: genuine",
    ]
    
    print(f"\n  Sample {i+1}:")
    print(f"    True: {class_names[true_label]} | Pred: {class_names[pred]}")
    print(f"    Entropy confidence: {entropy_conf:.4f}")
    print(f"    Perplexity confidence: {perp_conf:.4f}")
    print(f"    Perplexity: {perp:.4f}")

print('\n✓ SUCCESS: Full pipeline works!')
