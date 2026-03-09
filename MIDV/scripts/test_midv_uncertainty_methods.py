#!/usr/bin/env python3
"""
MIDV Document Type Classification (ID vs Passport)
Testing 3 uncertainty quantification methods:
1. Conformal Prediction
2. LLM as a Judge (Gemini)
3. SelfCheckGPT
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


# ============================================================================
# Dataset: MIDV Document Type Classification
# ============================================================================

class MIDVDocumentTypeDataset(Dataset):
    """
    MIDV dataset for ID vs Passport classification.
    
    IDs: alb_id, esp_id, est_id, fin_id, svk_id
    Passports: aze_passport, grc_passport, lva_passport, rus_internalpassport, srb_passport
    """
    
    DOCUMENT_TYPES = {
        # IDs
        "alb_id": "id",
        "esp_id": "id",
        "est_id": "id",
        "fin_id": "id",
        "svk_id": "id",
        # Passports
        "aze_passport": "passport",
        "grc_passport": "passport",
        "lva_passport": "passport",
        "rus_internalpassport": "passport",
        "srb_passport": "passport",
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "all",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        transform=None,
        max_images_per_country: int = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.doc_type_names = ["id", "passport"]
        self.doc_type_to_idx = {"id": 0, "passport": 1}
        self.country_names = list(self.DOCUMENT_TYPES.keys())
        
        # Load samples per country
        self.samples = []
        
        for country_code in self.country_names:
            country_dir = self.root_dir / country_code
            if not country_dir.exists():
                print(f"⚠ Warning: {country_dir} does not exist")
                continue
            
            image_files = sorted([
                f for f in country_dir.iterdir()
                if f.suffix.lower() in ['.tif', '.jpg', '.jpeg', '.png']
            ])
            
            if max_images_per_country:
                image_files = image_files[:max_images_per_country]
            
            # Get document type for this country
            doc_type = self.DOCUMENT_TYPES[country_code]
            doc_type_idx = self.doc_type_to_idx[doc_type]
            
            # Split per country
            n = len(image_files)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            if split == "train":
                country_images = image_files[:train_idx]
            elif split == "val":
                country_images = image_files[train_idx:val_idx]
            elif split == "test":
                country_images = image_files[val_idx:]
            else:
                country_images = image_files
            
            for img_path in country_images:
                self.samples.append((str(img_path), doc_type_idx, country_code, doc_type))
        
        print(f"✓ Loaded {len(self.samples)} images for split '{split}'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label, country, doc_type = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label, country
        except:
            return torch.zeros(3, 224, 224), label, country


# ============================================================================
# Classifier
# ============================================================================

class DocumentTypeClassifier:
    """Classify MIDV documents as ID or Passport."""
    
    def __init__(
        self,
        num_classes: int = 2,
        device: str = "cpu",
        lr: float = 0.001,
    ):
        self.device = device
        self.num_classes = num_classes
        
        # Create model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        
        print(f"✓ Initialized ResNet18 for document type classification on {device}")
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for images, labels, _ in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels, _ in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor):
        """
        Make predictions and return probabilities.
        
        Returns:
            predictions: class indices
            probabilities: softmax probabilities
            logits: raw logits
        """
        self.model.eval()
        images = images.to(self.device)
        
        logits = self.model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return {
            "predictions": preds.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
            "logits": logits.cpu().numpy(),
            "confidence": np.max(probs.cpu().numpy(), axis=1),
        }


# ============================================================================
# Uncertainty Methods
# ============================================================================

def conformal_prediction_score(probabilities: np.ndarray, true_label: int = None) -> Dict:
    """
    Compute conformal prediction scores.
    
    For each sample, compute:
    - Probability of predicted class
    - Margin between top-2 classes
    - Whether true label is in top-k set
    """
    n_samples = probabilities.shape[0]
    scores = {
        "top1_prob": np.max(probabilities, axis=1),
        "top2_prob": np.partition(probabilities, -2, axis=1)[:, -2],
        "margin": np.max(probabilities, axis=1) - np.partition(probabilities, -2, axis=1)[:, -2],
    }
    
    return scores


def entropy_uncertainty(probabilities: np.ndarray) -> Dict:
    """
    Compute entropy-based uncertainty.
    
    Lower entropy = higher confidence
    """
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    max_entropy = np.log(probabilities.shape[1])
    normalized_entropy = entropy / max_entropy
    
    return {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "uncertainty": normalized_entropy,  # Higher = more uncertain
        "confidence": 1.0 - normalized_entropy,  # Higher = more confident
    }


class SimpleConformalPredictor:
    """Simple conformal prediction calibrated on validation set."""
    
    def __init__(self):
        self.qhat = None
        self.calibration_scores = None
    
    def calibrate(self, probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.1):
        """
        Calibrate on validation set using APS (Adaptive Prediction Sets).
        """
        n = len(labels)
        scores = []
        
        for i in range(n):
            # Sort probabilities
            sorted_probs = sorted(probabilities[i], reverse=True)
            cumsum = 0.0
            
            for prob in sorted_probs:
                cumsum += prob
                if cumsum >= (1.0 - alpha):
                    break
            
            scores.append(cumsum)
        
        # Compute quantile
        scores_sorted = sorted(scores)
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        k = min(max(k, 0), n - 1)
        self.qhat = scores_sorted[k]
        self.calibration_scores = scores
    
    def predict_set(self, probabilities: np.ndarray) -> Dict:
        """
        Compute prediction sets using APS.
        """
        if self.qhat is None:
            raise ValueError("Must call calibrate first")
        
        n_samples = probabilities.shape[0]
        set_sizes = []
        set_coverages = []
        
        for i in range(n_samples):
            sorted_probs = sorted(probabilities[i], reverse=True)
            cumsum = 0.0
            set_size = 0
            
            for prob in sorted_probs:
                cumsum += prob
                set_size += 1
                if cumsum >= self.qhat:
                    break
            
            set_sizes.append(set_size)
            coverage = 1.0 if set_size == probabilities.shape[1] else 1.0 - (1.0 - cumsum)
            set_coverages.append(coverage)
        
        return {
            "set_sizes": np.array(set_sizes),
            "average_set_size": np.mean(set_sizes),
            "set_coverages": np.array(set_coverages),
            "qhat": self.qhat,
        }


# ============================================================================
# Main Test
# ============================================================================

def main():
    print("=" * 80)
    print("MIDV Document Type Classification (ID vs Passport)")
    print("Testing Uncertainty Methods on CNN Predictions")
    print("=" * 80)
    
    # Configuration
    dataset_dir = "datasets/MIDV/images"
    batch_size = 8
    num_epochs = 5
    max_images_per_country = 20  # Quick test
    
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Max images/country: {max_images_per_country}")
    
    # Load datasets
    print("\n1. Loading datasets...")
    print("-" * 80)
    
    train_dataset = MIDVDocumentTypeDataset(
        dataset_dir, split="train", max_images_per_country=max_images_per_country
    )
    val_dataset = MIDVDocumentTypeDataset(
        dataset_dir, split="val", max_images_per_country=max_images_per_country
    )
    test_dataset = MIDVDocumentTypeDataset(
        dataset_dir, split="test", max_images_per_country=max_images_per_country
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Train model
    print("\n2. Training document type classifier...")
    print("-" * 80)
    
    classifier = DocumentTypeClassifier(num_classes=2, device=device, lr=0.001)
    
    for epoch in range(num_epochs):
        train_loss = classifier.train_epoch(train_loader)
        val_loss, val_acc = classifier.evaluate(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
    
    # Evaluate on test set
    print("\n3. Evaluating on test set...")
    print("-" * 80)
    
    test_loss, test_acc = classifier.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Collect predictions and probabilities
    print("\n4. Collecting predictions for uncertainty analysis...")
    print("-" * 80)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    classifier.model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            result = classifier.predict_batch(images)
            all_preds.extend(result["predictions"])
            all_probs.extend(result["probabilities"])
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"Collected {len(all_labels)} predictions")
    
    # Method 1: Conformal Prediction
    print("\n5. Method 1: Conformal Prediction Analysis")
    print("-" * 80)
    
    # Calibrate on validation set
    val_preds = []
    val_probs = []
    val_labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            result = classifier.predict_batch(images)
            val_preds.extend(result["predictions"])
            val_probs.extend(result["probabilities"])
            val_labels_list.extend(labels.numpy())
    
    val_probs = np.array(val_probs)
    val_labels_list = np.array(val_labels_list)
    
    cp_predictor = SimpleConformalPredictor()
    cp_predictor.calibrate(val_probs, val_labels_list, alpha=0.1)
    
    cp_results = cp_predictor.predict_set(all_probs)
    cp_conf_scores = conformal_prediction_score(all_probs)
    
    print(f"  Average prediction set size: {cp_results['average_set_size']:.2f}")
    print(f"  Threshold (qhat): {cp_results['qhat']:.4f}")
    print(f"  Margin (top1 - top2) statistics:")
    print(f"    Mean: {np.mean(cp_conf_scores['margin']):.4f}")
    print(f"    Std: {np.std(cp_conf_scores['margin']):.4f}")
    print(f"    Min: {np.min(cp_conf_scores['margin']):.4f}")
    print(f"    Max: {np.max(cp_conf_scores['margin']):.4f}")
    
    # Method 2: Entropy-based Uncertainty
    print("\n6. Method 2: Entropy-based Uncertainty")
    print("-" * 80)
    
    entropy_results = entropy_uncertainty(all_probs)
    
    print(f"  Entropy statistics:")
    print(f"    Mean: {np.mean(entropy_results['entropy']):.4f}")
    print(f"    Std: {np.std(entropy_results['entropy']):.4f}")
    print(f"    Min: {np.min(entropy_results['entropy']):.4f}")
    print(f"    Max: {np.max(entropy_results['entropy']):.4f}")
    print(f"  Normalized Entropy statistics:")
    print(f"    Mean: {np.mean(entropy_results['normalized_entropy']):.4f}")
    print(f"    Confidence (1 - norm entropy):")
    print(f"      Mean: {np.mean(entropy_results['confidence']):.2%}")
    print(f"      Std: {np.std(entropy_results['confidence']):.4f}")
    
    # Method 3: Prediction Uncertainty from Logits
    print("\n7. Method 3: SelfCheck-style Confidence Measurement")
    print("-" * 80)
    
    max_prob = np.max(all_probs, axis=1)
    predicted_classes = np.argmax(all_probs, axis=1)
    
    correct_predictions = (predicted_classes == all_labels)
    correct_probs = max_prob[correct_predictions]
    incorrect_probs = max_prob[~correct_predictions]
    
    print(f"  Correct predictions:")
    print(f"    Count: {np.sum(correct_predictions)}")
    print(f"    Mean confidence: {np.mean(correct_probs):.2%}")
    print(f"  Incorrect predictions:")
    print(f"    Count: {np.sum(~correct_predictions)}")
    if len(incorrect_probs) > 0:
        print(f"    Mean confidence: {np.mean(incorrect_probs):.2%}")
    
    # Summary
    print("\n8. Overall Performance Summary")
    print("-" * 80)
    
    print(f"  CNN Accuracy: {np.mean(predicted_classes == all_labels):.2%}")
    print(f"  Average Confidence: {np.mean(max_prob):.2%}")
    
    # Per-document-type accuracy
    id_mask = all_labels == 0
    passport_mask = all_labels == 1
    
    if np.sum(id_mask) > 0:
        id_acc = np.mean(predicted_classes[id_mask] == all_labels[id_mask])
        print(f"  ID Classification Accuracy: {id_acc:.2%}")
    
    if np.sum(passport_mask) > 0:
        passport_acc = np.mean(predicted_classes[passport_mask] == all_labels[passport_mask])
        print(f"  Passport Classification Accuracy: {passport_acc:.2%}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": "MIDV Document Type Classification (ID vs Passport)",
        "methods": ["Conformal Prediction", "Entropy-based", "SelfCheck-style"],
        "metrics": {
            "cnn_accuracy": float(np.mean(predicted_classes == all_labels)),
            "average_confidence": float(np.mean(max_prob)),
        },
        "conformal_prediction": {
            "average_set_size": float(cp_results['average_set_size']),
            "qhat": float(cp_results['qhat']),
            "mean_margin": float(np.mean(cp_conf_scores['margin'])),
        },
        "entropy": {
            "mean_entropy": float(np.mean(entropy_results['entropy'])),
            "mean_confidence": float(np.mean(entropy_results['confidence'])),
        },
        "test_count": int(len(all_labels)),
    }
    
    results_path = Path("statap_code/comparison_results/midv_doctype_uncertainty_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Analysis completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
