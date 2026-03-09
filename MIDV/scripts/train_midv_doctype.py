#!/usr/bin/env python3
"""
Train MIDV Document Type Classifier (ID vs Passport)
This script trains a ResNet18 model that can then be used with uncertainty methods.
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
import time


class MIDVDocTypeDataset(Dataset):
    """ID vs Passport classification dataset."""
    
    DOCUMENT_TYPES = {
        # IDs (class 0)
        "alb_id": 0,
        "esp_id": 0,
        "est_id": 0,
        "fin_id": 0,
        "svk_id": 0,
        # Passports (class 1)
        "aze_passport": 1,
        "grc_passport": 1,
        "lva_passport": 1,
        "rus_internalpassport": 1,
        "srb_passport": 1,
    }
    
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
        
        for country, label in self.DOCUMENT_TYPES.items():
            country_dir = self.root_dir / country
            if not country_dir.exists():
                continue
            
            images = sorted([f for f in country_dir.iterdir() 
                           if f.suffix.lower() in ['.tif', '.jpg', '.jpeg', '.png']])
            
            if max_images:
                images = images[:max_images]
            
            n = len(images)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            if split == "train":
                images = images[:train_idx]
            elif split == "val":
                images = images[train_idx:val_idx]
            elif split == "test":
                images = images[val_idx:]
            
            for img_path in images:
                self.samples.append((str(img_path), label))
        
        print(f"✓ Loaded {len(self.samples)} images for split '{split}'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img), label
        except:
            return torch.zeros(3, 224, 224), label


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
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
    for images, labels in dataloader:
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
    print("Training MIDV Document Type Classifier (ID vs Passport)")
    print("=" * 80)
    
    # Configuration
    dataset_dir = "datasets/MIDV/images"
    batch_size = 16
    num_epochs = 3  # Reduced for quick test
    max_images_per_country = 25  # Reduced
    
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Max images/country: {max_images_per_country}")
    
    # Create datasets
    print("\n1. Loading datasets...")
    print("-" * 80)
    
    train_dataset = MIDVDocTypeDataset(
        dataset_dir, split="train", max_images=max_images_per_country
    )
    val_dataset = MIDVDocTypeDataset(
        dataset_dir, split="val", max_images=max_images_per_country
    )
    test_dataset = MIDVDocTypeDataset(
        dataset_dir, split="test", max_images=max_images_per_country
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print("\n2. Creating model...")
    print("-" * 80)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: ID, Passport
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    print("✓ ResNet18 ready for training")
    
    # Train
    print("\n3. Training...")
    print("-" * 80)
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        scheduler.step()
    
    train_time = time.time() - start_time
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    print("-" * 80)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Training time: {train_time:.1f}s")
    
    # Save model
    print("\n5. Saving model...")
    print("-" * 80)
    
    model_path = Path("statap_code/comparison_results/midv_doctype_model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "test_accuracy": test_acc,
        "architecture": "ResNet18",
        "num_classes": 2,
        "class_names": ["ID", "Passport"],
    }, model_path)
    
    print(f"✓ Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print("✓ Training completed!")
    print("=" * 80)
    print("\nYou can now use this model with:")
    print("  python test_midv_uncertainty_methods.py")
    print("  python test_midv_cnn_vs_llm.py")


if __name__ == "__main__":
    main()
