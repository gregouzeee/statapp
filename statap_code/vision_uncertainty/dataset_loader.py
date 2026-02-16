"""Dataset loader for fraud ID dataset (EST)."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple


class FraudIDDataset(Dataset):
    """
    Load EST fraud ID dataset.
    Structure:
        EST/
        ├── fraud1_copy_and_move/
        ├── fraud2_face_morphing/
        ├── fraud3_face_replacement/
        ├── fraud4_combined/
        ├── fraud5_inpaint_and_rewrite/
        ├── fraud6_crop_and_replace/
        └── positive/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "all",  # 'all', 'train', 'val', 'test'
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        transform=None,
    ):
        """
        Args:
            root_dir: path to EST directory
            split: data split
            train_ratio, val_ratio: ratios for train/val/test
            transform: optional torchvision transforms
        """
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
        
        # Define class names and labels
        self.fraud_types = [
            "fraud1_copy_and_move",
            "fraud2_face_morphing",
            "fraud3_face_replacement",
            "fraud4_combined",
            "fraud5_inpaint_and_rewrite",
            "fraud6_crop_and_replace",
        ]
        self.class_names = self.fraud_types + ["positive"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load file paths and labels
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            image_files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            for img_path in image_files:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        # Split dataset
        n = len(self.samples)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        if split == "train":
            self.samples = self.samples[:train_idx]
        elif split == "val":
            self.samples = self.samples[train_idx:val_idx]
        elif split == "test":
            self.samples = self.samples[val_idx:]
        # else: split == "all", keep all
        
        print(f"Loaded {len(self.samples)} samples for split '{split}'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: (3, 224, 224)
            label: class index (0-6 for fraud types, 6 for positive)
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        root_dir: path to EST directory
        batch_size: batch size
        num_workers: number of workers for dataloader
        train_ratio, val_ratio: split ratios
    
    Returns:
        dict of dataloaders for 'train', 'val', 'test'
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = FraudIDDataset(
            root_dir=root_dir,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
        )
    
    return dataloaders
