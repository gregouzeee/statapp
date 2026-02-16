"""Example: Test fraud ID classifier with uncertainty quantification."""

import os
import sys
import torch
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from statap_code.vision_uncertainty.fraud_classifier import FraudIDClassifier
from statap_code.vision_uncertainty.dataset_loader import create_dataloaders


def main():
    # Configuration
    est_dataset_path = "/Users/darfilalikenan/Downloads/EST"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n=== Loading Dataset ===")
    dataloaders = create_dataloaders(
        root_dir=est_dataset_path,
        batch_size=batch_size,
        num_workers=0,
    )
    
    # Initialize classifier
    print("\n=== Initializing Classifier ===")
    classifier = FraudIDClassifier(
        model_name="resnet50",
        num_classes=7,
        device=device,
    )
    
    # Calibrate conformal prediction on validation set
    print("\n=== Calibrating Conformal Prediction ===")
    classifier.calibrate_conformal(dataloaders['val'])
    
    # Evaluate on test set
    print("\n=== Evaluating on Test Set ===")
    metrics = classifier.evaluate(dataloaders['test'], return_uncertainties=True)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Demonstrate prediction with uncertainties
    print("\n=== Sample Predictions ===")
    images, labels = next(iter(dataloaders['test']))
    result = classifier.predict_batch(images[:4], return_uncertainties=True)
    
    class_names = [
        "fraud1: copy_and_move",
        "fraud2: face_morphing",
        "fraud3: face_replacement",
        "fraud4: combined",
        "fraud5: inpaint_rewrite",
        "fraud6: crop_replace",
        "positive: genuine",
    ]
    
    for i in range(min(4, len(labels))):
        pred = result['predictions'][i]
        true_label = labels[i].item()
        entropy = result['entropy'][i]
        log_prob = result['log_prob'][i]
        perp_conf = result['perplexity_confidence'][i]
        
        print(f"\nSample {i+1}:")
        print(f"  True: {class_names[true_label]}")
        print(f"  Pred: {class_names[pred]} (correct: {pred == true_label})")
        print(f"  Entropy: {entropy:.4f} | Log-prob: {log_prob:.4f} | Perp-conf: {perp_conf:.4f}")
        
        if 'conformal_set_size' in result:
            set_size = result['conformal_set_size'][i]
            efficiency = result['conformal_efficiency'][i]
            print(f"  Conformal set size: {set_size} / 7 | Efficiency: {efficiency:.4f}")


if __name__ == '__main__':
    main()
