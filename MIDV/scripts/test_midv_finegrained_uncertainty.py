#!/usr/bin/env python3
"""
MIDV Fine-Grained Country Classification (10 classes)
Testing 3 uncertainty quantification methods on HARD task:
1. Conformal Prediction
2. Entropy-based Uncertainty
3. SelfCheckGPT-style validation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report


class MIDVCountryDataset(Dataset):
    """Fine-grained: 10-class country classification dataset."""
    
    DOCUMENT_TYPES = {
        "alb_id": 0, "esp_id": 1, "est_id": 2, "fin_id": 3, "svk_id": 4,
        "aze_passport": 5, "grc_passport": 6, "lva_passport": 7,
        "rus_internalpassport": 8, "srb_passport": 9,
    }
    
    COUNTRY_NAMES = {
        0: "Albanian ID", 1: "Spanish ID", 2: "Estonian ID", 3: "Finnish ID",
        4: "Slovak ID", 5: "Azerbaijani Passport", 6: "Greek Passport",
        7: "Latvian Passport", 8: "Russian Passport", 9: "Serbian Passport",
    }
    
    NUM_CLASSES = 10
    
    def __init__(self, root_dir: str, split: str = "all", train_ratio: float = 0.7,
                 val_ratio: float = 0.15, max_images: int = None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.samples = []
        
        for country_code, label in self.DOCUMENT_TYPES.items():
            country_dir = self.root_dir / country_code
            if not country_dir.exists():
                continue
            
            image_files = sorted([
                f for f in country_dir.iterdir()
                if f.suffix.lower() in ['.tif', '.jpg', '.jpeg', '.png']
            ])
            
            if max_images:
                image_files = image_files[:max_images]
            
            n = len(image_files)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            for i, img_file in enumerate(image_files):
                if i < train_idx:
                    file_split = "train"
                elif i < val_idx:
                    file_split = "val"
                else:
                    file_split = "test"
                
                self.samples.append({
                    "image_path": img_file,
                    "label": label,
                    "country": country_code,
                    "split": file_split,
                })
        
        self.samples = [s for s in self.samples if s["split"] == split or split == "all"]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, sample["label"], sample["country"]


class FineGrainedUncertainty:
    """Combine all 3 uncertainty methods for 10-class classification."""
    
    @staticmethod
    def conformal_score(probs: np.ndarray) -> np.ndarray:
        """Conformal prediction: margin between top-2 classes."""
        top1 = np.max(probs, axis=1)
        top2 = np.partition(probs, -2, axis=1)[:, -2]
        margin = top1 - top2
        return margin
    
    @staticmethod
    def entropy_confidence(probs: np.ndarray) -> np.ndarray:
        """Entropy-based confidence: 1 - normalized_entropy."""
        n_classes = probs.shape[1]
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy
        confidence = 1.0 - normalized_entropy
        return confidence
    
    @staticmethod
    def selfcheck_confidence(probs: np.ndarray) -> np.ndarray:
        """SelfCheck-style: maximum softmax probability."""
        return np.max(probs, axis=1)
    
    @staticmethod
    def ensemble_vote(probs: np.ndarray, weights: List[float] = None) -> Dict:
        """Ensemble combines all 3 methods."""
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        
        conf_score = FineGrainedUncertainty.conformal_score(probs)
        ent_score = FineGrainedUncertainty.entropy_confidence(probs)
        selfcheck_score = FineGrainedUncertainty.selfcheck_confidence(probs)
        
        ensemble_conf = (
            weights[0] * conf_score +
            weights[1] * ent_score +
            weights[2] * selfcheck_score
        )
        
        return {
            "ensemble_confidence": ensemble_conf,
            "conformal_score": conf_score,
            "entropy_confidence": ent_score,
            "selfcheck_confidence": selfcheck_score,
        }


def main():
    print("=" * 80)
    print("MIDV FINE-GRAINED - UNCERTAINTY METHODS TESTING (10 classes) — PHOTO")
    print("Photos = real acquisition conditions → expect genuine uncertainty")
    print("=" * 80)
    
    # Configuration
    dataset_dir = Path("MIDV/datasets/MIDV/photo/images")
    model_path = Path("MIDV/results/midv_finegrained_photo_model.pt")
    batch_size = 16
    max_images = 100
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found at {dataset_dir}")
        return
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print(f"   Run: python MIDV/scripts/train_midv_finegrained.py")
        return
    
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load model
    print(f"\n1️⃣  Loading model from {model_path}...")
    print("-" * 80)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    # Load test dataset
    print(f"\n2️⃣  Loading test dataset...")
    print("-" * 80)
    
    test_dataset = MIDVCountryDataset(
        str(dataset_dir), split="test", max_images=max_images
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Test set: {len(test_dataset)} images")
    
    # Test predictions
    print(f"\n3️⃣  Computing predictions and uncertainty scores...")
    print("-" * 80)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    all_countries = []
    
    with torch.no_grad():
        for images, labels, countries in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_countries.extend(countries)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy
    accuracy = np.mean(all_predictions == all_labels)
    print(f"✓ Classification Accuracy: {accuracy*100:.2f}%")
    
    # Uncertainty scores
    print(f"\n4️⃣  Computing ensemble uncertainty...")
    print("-" * 80)
    
    ensemble_results = FineGrainedUncertainty.ensemble_vote(all_probs)
    
    mean_conf = np.mean(ensemble_results["ensemble_confidence"])
    min_conf = np.min(ensemble_results["ensemble_confidence"])
    max_conf = np.max(ensemble_results["ensemble_confidence"])
    
    print(f"  Mean ensemble confidence: {mean_conf*100:.2f}%")
    print(f"  Min confidence: {min_conf*100:.2f}%")
    print(f"  Max confidence: {max_conf*100:.2f}%")
    
    # Per-method stats
    print(f"\n5️⃣  Per-method statistics:")
    print("-" * 80)
    
    methods = {
        "Conformal": ensemble_results["conformal_score"],
        "Entropy": ensemble_results["entropy_confidence"],
        "SelfCheck": ensemble_results["selfcheck_confidence"],
    }
    
    for method_name, scores in methods.items():
        print(f"  {method_name:12s}: mean={np.mean(scores)*100:.2f}%, "
              f"std={np.std(scores)*100:.2f}%")
    
    # Calibration analysis
    print(f"\n6️⃣  Calibration Analysis:")
    print("-" * 80)
    
    correct_mask = all_predictions == all_labels
    if np.any(correct_mask):
        conf_correct = np.mean(ensemble_results["ensemble_confidence"][correct_mask])
    else:
        conf_correct = 0
    
    if np.any(~correct_mask):
        conf_wrong = np.mean(ensemble_results["ensemble_confidence"][~correct_mask])
    else:
        conf_wrong = 0
    
    print(f"  Avg confidence when CORRECT: {conf_correct*100:.2f}%")
    print(f"  Avg confidence when WRONG:   {conf_wrong*100:.2f}%")
    print(f"  Calibration gap: {(conf_correct - conf_wrong)*100:.2f}%")
    
    # Rejection strategy
    print(f"\n7️⃣  Rejection Thresholds:")
    print("-" * 80)
    
    thresholds = [0.5, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        accepted_mask = ensemble_results["ensemble_confidence"] >= threshold
        if np.sum(accepted_mask) > 0:
            accuracy_accepted = np.sum(correct_mask[accepted_mask]) / np.sum(accepted_mask)
            coverage = 100 * np.sum(accepted_mask) / len(all_predictions)
            print(f"  Threshold {threshold:.0%}: Coverage={coverage:5.1f}%, Accuracy={accuracy_accepted*100:5.1f}%")
    
    # Confusion on hard pairs
    print(f"\n8️⃣  Hardest Country Pairs (Confusions):")
    print("-" * 80)
    
    confusion = {}
    for true, pred, country in zip(all_labels, all_predictions, all_countries):
        if true != pred:
            true_name = MIDVCountryDataset.COUNTRY_NAMES[true]
            pred_name = MIDVCountryDataset.COUNTRY_NAMES[pred]
            pair = f"{true_name} → {pred_name}"
            confusion[pair] = confusion.get(pair, 0) + 1
    
    if confusion:
        sorted_pairs = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs[:5]:
            print(f"  {pair}: {count} times")
    else:
        print(f"  ✓ No confusions! (Perfect accuracy)")
    
    # Save results
    print(f"\n9️⃣  Saving results...")
    print("-" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": "MIDV Fine-Grained Country Classification (10 classes) — PHOTO - Uncertainty",
        "metrics": {
            "accuracy": float(accuracy),
            "mean_confidence": float(mean_conf),
            "min_confidence": float(min_conf),
            "max_confidence": float(max_conf),
            "total_samples": len(all_labels),
            "correct_predictions": int(np.sum(correct_mask)),
        },
        "calibration": {
            "conf_correct": float(conf_correct),
            "conf_wrong": float(conf_wrong),
            "calibration_gap": float(conf_correct - conf_wrong),
        },
        "per_method": {
            "conformal": {
                "mean": float(np.mean(ensemble_results["conformal_score"])),
                "std": float(np.std(ensemble_results["conformal_score"])),
            },
            "entropy": {
                "mean": float(np.mean(ensemble_results["entropy_confidence"])),
                "std": float(np.std(ensemble_results["entropy_confidence"])),
            },
            "selfcheck": {
                "mean": float(np.mean(ensemble_results["selfcheck_confidence"])),
                "std": float(np.std(ensemble_results["selfcheck_confidence"])),
            },
        },
        "confusion_pairs": dict(sorted_pairs[:5]) if confusion else {},
    }
    
    results_path = Path("MIDV/results/midv_finegrained_photo_uncertainty_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Fine-grained uncertainty testing completed!")
    print("=" * 80)
    print(f"\n📊 KEY FINDINGS:")
    print(f"   • This is a HARD task (10-class) vs easy binary classification")
    print(f"   • Real uncertainty: {mean_conf*100:.0f}% avg (should be 50-80% for hard tasks)")
    print(f"   • Calibration gap: {(conf_correct - conf_wrong)*100:.1f}% (should be large)")
    print(f"   • Confused country pairs: {len(confusion)} (which countries are confused?)")
    print()


if __name__ == "__main__":
    main()
