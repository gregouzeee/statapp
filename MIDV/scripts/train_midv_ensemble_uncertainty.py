#!/usr/bin/env python3
"""
MIDV Document Type Classification - Ensemble Uncertainty Method
Train on FULL dataset and test ensemble combining:
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
import time


# ============================================================================
# Dataset: MIDV Document Type Classification
# ============================================================================

class MIDVDocumentTypeDataset(Dataset):
    """MIDV dataset for ID vs Passport classification (Full Dataset)."""
    
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
        max_images_per_country: int = None,  # None = use all
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
        
        self.samples = []
        
        for country_code in self.country_names:
            country_dir = self.root_dir / country_code
            if not country_dir.exists():
                continue
            
            image_files = sorted([
                f for f in country_dir.iterdir()
                if f.suffix.lower() in ['.tif', '.jpg', '.jpeg', '.png']
            ])
            
            if max_images_per_country:
                image_files = image_files[:max_images_per_country]
            
            doc_type = self.DOCUMENT_TYPES[country_code]
            doc_type_idx = self.doc_type_to_idx[doc_type]
            
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
                    "label": doc_type_idx,
                    "doc_type": doc_type,
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


# ============================================================================
# Model: ResNet18 for Document Classification
# ============================================================================

class DocumentClassifier:
    def __init__(self, num_classes: int = 2, lr: float = 0.001):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if not torch.backends.mps.is_available():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        
        print(f"✓ Initialized ResNet18 on {device}")
    
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
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Model loaded from {path}")


# ============================================================================
# Ensemble Uncertainty Methods
# ============================================================================

class EnsembleUncertainty:
    """Combine all 3 uncertainty methods into one ensemble vote."""
    
    @staticmethod
    def conformal_score(probs: np.ndarray) -> np.ndarray:
        """
        Conformal prediction confidence: margin between top-2 classes.
        Range: [0, 1], higher = more confident
        """
        if probs.shape[1] == 1:
            return np.ones(probs.shape[0])
        
        top1 = np.max(probs, axis=1)
        top2 = np.partition(probs, -2, axis=1)[:, -2]
        margin = top1 - top2
        return margin
    
    @staticmethod
    def entropy_confidence(probs: np.ndarray) -> np.ndarray:
        """
        Entropy-based confidence: 1 - normalized_entropy.
        Range: [0, 1], higher = more confident
        """
        n_classes = probs.shape[1]
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy
        confidence = 1.0 - normalized_entropy
        return confidence
    
    @staticmethod
    def selfcheck_confidence(probs: np.ndarray) -> np.ndarray:
        """
        SelfCheck-style: maximum softmax probability.
        Range: [0, 1], higher = more confident
        """
        return np.max(probs, axis=1)
    
    @staticmethod
    def ensemble_vote(probs: np.ndarray, weights: List[float] = None) -> Dict:
        """
        Ensemble combines all 3 methods.
        
        Returns:
            - ensemble_confidence: weighted average of all 3 methods
            - method_votes: individual scores from each method
            - decision: "high_confidence" or "low_confidence"
        """
        if weights is None:
            weights = [1/3, 1/3, 1/3]  # Equal weights
        
        conf_score = EnsembleUncertainty.conformal_score(probs)
        ent_score = EnsembleUncertainty.entropy_confidence(probs)
        selfcheck_score = EnsembleUncertainty.selfcheck_confidence(probs)
        
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
            "decision": np.where(ensemble_conf > 0.8, "high_confidence", "low_confidence"),
        }


# ============================================================================
# Main Training & Testing
# ============================================================================

def main():
    print("=" * 80)
    print("MIDV ENSEMBLE UNCERTAINTY - FULL DATASET TRAINING")
    print("=" * 80)
    
    # Configuration
    DATASET_PATH = Path("datasets/MIDV/images")
    MODEL_PATH = Path("statap_code/comparison_results/midv_ensemble_model.pt")
    RESULTS_PATH = Path("statap_code/comparison_results/midv_ensemble_uncertainty_results.json")
    
    NUM_EPOCHS = 5
    BATCH_SIZE = 16
    MAX_IMAGES = None  # Use ALL images (100 per country)
    
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return
    
    print(f"\n📊 Loading FULL dataset (max_images={MAX_IMAGES})...")
    
    # Load datasets
    train_dataset = MIDVDocumentTypeDataset(
        root_dir=str(DATASET_PATH),
        split="train",
        max_images_per_country=MAX_IMAGES,
    )
    val_dataset = MIDVDocumentTypeDataset(
        root_dir=str(DATASET_PATH),
        split="val",
        max_images_per_country=MAX_IMAGES,
    )
    test_dataset = MIDVDocumentTypeDataset(
        root_dir=str(DATASET_PATH),
        split="test",
        max_images_per_country=MAX_IMAGES,
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print(f"\n🤖 Initializing ResNet18 classifier...")
    classifier = DocumentClassifier(num_classes=2, lr=0.001)
    
    # Training loop
    print(f"\n📈 Training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        train_loss = classifier.train_epoch(train_loader)
        val_loss, val_acc = classifier.evaluate(val_loader)
        classifier.scheduler.step()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f}s")
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(MODEL_PATH))
    
    # Test with ensemble uncertainty
    print(f"\n🧪 Testing ensemble uncertainty methods on test set...")
    
    all_predictions = []
    all_labels = []
    all_probs = []
    all_countries = []
    
    classifier.model.eval()
    with torch.no_grad():
        for images, labels, countries in test_loader:
            result = classifier.predict_batch(images)
            
            all_predictions.extend(result["predictions"])
            all_labels.extend(labels.numpy())
            all_probs.extend(result["probabilities"])
            all_countries.extend(countries)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Accuracy
    accuracy = np.mean(all_predictions == all_labels)
    print(f"✓ CNN Accuracy: {accuracy*100:.2f}%")
    
    # Ensemble uncertainty
    print(f"\n📊 Computing ensemble uncertainty scores...")
    ensemble_results = EnsembleUncertainty.ensemble_vote(all_probs, weights=[1/3, 1/3, 1/3])
    
    ensemble_conf = ensemble_results["ensemble_confidence"]
    high_conf_count = np.sum(ensemble_results["decision"] == "high_confidence")
    
    print(f"  Mean ensemble confidence: {np.mean(ensemble_conf)*100:.2f}%")
    print(f"  High confidence samples: {high_conf_count}/{len(all_predictions)}")
    print(f"  Min confidence: {np.min(ensemble_conf)*100:.2f}%")
    print(f"  Max confidence: {np.max(ensemble_conf)*100:.2f}%")
    
    # Per-method statistics
    print(f"\n📈 Per-method statistics:")
    methods = {
        "Conformal": ensemble_results["conformal_score"],
        "Entropy": ensemble_results["entropy_confidence"],
        "SelfCheck": ensemble_results["selfcheck_confidence"],
    }
    
    for name, scores in methods.items():
        print(f"  {name}:")
        print(f"    Mean: {np.mean(scores)*100:.2f}%")
        print(f"    Std: {np.std(scores)*100:.2f}%")
        print(f"    Min: {np.min(scores)*100:.2f}%")
        print(f"    Max: {np.max(scores)*100:.2f}%")
    
    # Per-class analysis
    print(f"\n📊 Per-class analysis:")
    for class_idx, class_name in enumerate(["ID", "Passport"]):
        mask = all_labels == class_idx
        class_acc = np.mean(all_predictions[mask] == all_labels[mask])
        class_conf = np.mean(ensemble_conf[mask])
        
        print(f"  {class_name}:")
        print(f"    Accuracy: {class_acc*100:.2f}%")
        print(f"    Mean confidence: {class_conf*100:.2f}%")
        print(f"    Samples: {np.sum(mask)}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_images_per_country": MAX_IMAGES,
            "training_time_seconds": training_time,
            "dataset_sizes": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            }
        },
        "metrics": {
            "cnn_accuracy": float(accuracy),
            "mean_ensemble_confidence": float(np.mean(ensemble_conf)),
            "high_confidence_samples": int(high_conf_count),
            "total_samples": len(all_predictions),
        },
        "per_method": {
            "conformal": {
                "mean": float(np.mean(ensemble_results["conformal_score"])),
                "std": float(np.std(ensemble_results["conformal_score"])),
                "min": float(np.min(ensemble_results["conformal_score"])),
                "max": float(np.max(ensemble_results["conformal_score"])),
            },
            "entropy": {
                "mean": float(np.mean(ensemble_results["entropy_confidence"])),
                "std": float(np.std(ensemble_results["entropy_confidence"])),
                "min": float(np.min(ensemble_results["entropy_confidence"])),
                "max": float(np.max(ensemble_results["entropy_confidence"])),
            },
            "selfcheck": {
                "mean": float(np.mean(ensemble_results["selfcheck_confidence"])),
                "std": float(np.std(ensemble_results["selfcheck_confidence"])),
                "min": float(np.min(ensemble_results["selfcheck_confidence"])),
                "max": float(np.max(ensemble_results["selfcheck_confidence"])),
            },
        },
        "per_class": {
            "id": {
                "accuracy": float(np.mean(all_predictions[all_labels == 0] == all_labels[all_labels == 0])) if np.any(all_labels == 0) else 0.0,
                "mean_confidence": float(np.mean(ensemble_conf[all_labels == 0])) if np.any(all_labels == 0) else 0.0,
                "samples": int(np.sum(all_labels == 0)),
            },
            "passport": {
                "accuracy": float(np.mean(all_predictions[all_labels == 1] == all_labels[all_labels == 1])) if np.any(all_labels == 1) else 0.0,
                "mean_confidence": float(np.mean(ensemble_conf[all_labels == 1])) if np.any(all_labels == 1) else 0.0,
                "samples": int(np.sum(all_labels == 1)),
            },
        },
    }
    
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {RESULTS_PATH}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
