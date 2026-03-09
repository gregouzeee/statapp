#!/usr/bin/env python3
"""
Train MIDV Fine-Grained Country Classifier (10 classes)
Replaces binary ID vs Passport classification with exact country identification.
This enables real uncertainty quantification testing on a HARD task.
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
from typing import Dict, Tuple
import json
from datetime import datetime
import time


class MIDVCountryDataset(Dataset):
    """Fine-grained: 10-class country classification dataset."""
    
    DOCUMENT_TYPES = {
        # IDs (5 classes)
        "alb_id": 0,
        "esp_id": 1,
        "est_id": 2,
        "fin_id": 3,
        "svk_id": 4,
        # Passports (5 classes)
        "aze_passport": 5,
        "grc_passport": 6,
        "lva_passport": 7,
        "rus_internalpassport": 8,
        "srb_passport": 9,
    }
    
    COUNTRY_NAMES = {
        0: "Albanian ID",
        1: "Spanish ID",
        2: "Estonian ID",
        3: "Finnish ID",
        4: "Slovak ID",
        5: "Azerbaijani Passport",
        6: "Greek Passport",
        7: "Latvian Passport",
        8: "Russian Passport",
        9: "Serbian Passport",
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


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def main():
    print("=" * 80)
    print("Training MIDV Fine-Grained Country Classifier (10 classes) — PHOTO dataset")
    print("Photos = perspective + lighting variations → HARDER than flat scans")
    print("=" * 80)
    
    # Configuration
    dataset_dir = Path("MIDV/datasets/MIDV/photo/images")
    batch_size = 16
    num_epochs = 5
    max_images_per_country = 100  # Use all images
    
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nDevice: {device}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Epochs: {num_epochs}", flush=True)
    print(f"Max images/country: {max_images_per_country}", flush=True)
    
    # Create datasets
    print("\n1. Loading datasets...")
    print("-" * 80)
    
    train_dataset = MIDVCountryDataset(
        str(dataset_dir), split="train", max_images=max_images_per_country
    )
    val_dataset = MIDVCountryDataset(
        str(dataset_dir), split="val", max_images=max_images_per_country
    )
    test_dataset = MIDVCountryDataset(
        str(dataset_dir), split="test", max_images=max_images_per_country
    )
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model with 10 output classes
    print("\n2. Initializing ResNet18 (10 classes)...")
    print("-" * 80)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    print(f"✓ ResNet18 initialized with 10 output classes")
    
    # Training loop
    print(f"\n3. Training for {num_epochs} epochs...")
    print("-" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%", flush=True)
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.1f}s")
    
    # Test evaluation
    print(f"\n4. Evaluating on test set...")
    print("-" * 80)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    
    # Save model
    print(f"\n5. Saving model...")
    print("-" * 80)
    
    model_dir = Path("MIDV/results")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "midv_finegrained_photo_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": "MIDV Fine-Grained Country Classification (10 classes) — PHOTO",
        "configuration": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "max_images_per_country": max_images_per_country,
            "num_classes": 10,
            "device": device,
        },
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "training_time_seconds": training_time,
        },
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
            "total": len(train_dataset) + len(val_dataset) + len(test_dataset),
        },
        "country_classes": MIDVCountryDataset.COUNTRY_NAMES,
    }
    
    results_path = model_dir / "midv_finegrained_photo_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("✓ Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
