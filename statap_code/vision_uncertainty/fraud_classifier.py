"""Main pipeline for fraud ID classification with uncertainty quantification."""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from typing import Dict, Tuple, List
import json
from pathlib import Path

from .entropy_uncertainty import EntropyUncertainty
from .perplexity_uncertainty import PerplexityUncertainty
from .conformal_prediction import ConformalPrediction
from .dataset_loader import FraudIDDataset, create_dataloaders


class FraudIDClassifier:
    """
    Complete pipeline for fraud ID classification with 3 uncertainty methods.
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained: bool = True,
    ):
        """
        Args:
            model_name: pretrained model name (resnet50, resnet101, etc.)
            num_classes: number of output classes
            device: 'cuda' or 'cpu'
            pretrained: whether to load pretrained ImageNet weights
        """
        self.device = device
        self.num_classes = num_classes
        
        # Load model
        if model_name.startswith("resnet"):
            self.model = getattr(models, model_name)(pretrained=pretrained)
            # Modify final layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name.startswith("mobilenet"):
            # mobilenet_v2 has classifier[1] as linear layer
            self.model = getattr(models, model_name)(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features if hasattr(self.model.classifier[1], 'in_features') else getattr(self.model.classifier[1], 'in_channels', None)
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize uncertainty methods
        self.entropy_method = EntropyUncertainty(temperature=1.0)
        self.perplexity_method = PerplexityUncertainty(temperature=1.0)
        self.conformal_method = ConformalPrediction(confidence_level=0.9)
        
        print(f"Initialized FraudIDClassifier with {model_name} on {device}")
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
        return_uncertainties: bool = True,
    ) -> Dict:
        """
        Predict on a batch of images.
        
        Args:
            images: shape (batch, 3, 224, 224)
            return_uncertainties: whether to compute uncertainty estimates
        
        Returns:
            dict with predictions, confidences, uncertainties
        """
        images = images.to(self.device)
        logits = self.model(images)
        
        result = {
            'logits': logits.cpu().numpy(),
            'predictions': logits.argmax(dim=-1).cpu().numpy(),
            'pred_probs': torch.nn.functional.softmax(logits, dim=-1).cpu().numpy(),
        }
        
        if return_uncertainties:
            # Entropy-based
            entropy, probs = self.entropy_method.get_uncertainty(logits)
            entropy_conf = self.entropy_method.get_calibrated_confidence(logits)
            result['entropy'] = entropy.cpu().numpy()
            result['entropy_confidence'] = entropy_conf.cpu().numpy()
            
            # Perplexity-based
            log_prob, perplexity = self.perplexity_method.get_uncertainty(logits)
            perp_conf = self.perplexity_method.get_confidence(logits)
            result['log_prob'] = log_prob.cpu().numpy()
            result['perplexity'] = perplexity.cpu().numpy()
            result['perplexity_confidence'] = perp_conf.cpu().numpy()
            
            # Conformal (needs calibration first)
            if self.conformal_method.quantile_val is not None:
                pred_sets, set_sizes = self.conformal_method.get_prediction_set(logits)
                efficiency, _ = self.conformal_method.get_efficiency_measure(logits)
                result['conformal_set_size'] = set_sizes.cpu().numpy()
                result['conformal_efficiency'] = efficiency.cpu().numpy()
                result['conformal_pred_sets'] = pred_sets.cpu().numpy()
        
        return result
    
    def calibrate_conformal(
        self,
        dataloader,
    ):
        """
        Calibrate conformal prediction on validation set.
        
        Args:
            dataloader: validation dataloader
        """
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                logits = self.model(images)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)
        
        self.conformal_method.calibrate(all_logits, all_labels)
        print(f"Conformal prediction calibrated with quantile = {self.conformal_method.quantile_val:.4f}")
    
    def evaluate(
        self,
        dataloader,
        return_uncertainties: bool = True,
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: data loader
            return_uncertainties: compute uncertainty metrics
        
        Returns:
            dict with accuracy and uncertainty metrics
        """
        all_preds = []
        all_labels = []
        all_entropies = []
        all_log_probs = []
        all_conformal_efficiencies = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                result = self.predict_batch(images, return_uncertainties=return_uncertainties)
                all_preds.append(result['predictions'])
                all_labels.append(labels.numpy())
                
                if return_uncertainties:
                    all_entropies.append(result['entropy'])
                    all_log_probs.append(result['log_prob'])
                    if 'conformal_efficiency' in result:
                        all_conformal_efficiencies.append(result['conformal_efficiency'])
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        accuracy = (all_preds == all_labels).mean()
        
        metrics = {'accuracy': accuracy}
        
        if return_uncertainties:
            all_entropies = np.concatenate(all_entropies)
            all_log_probs = np.concatenate(all_log_probs)
            
            metrics.update({
                'mean_entropy': all_entropies.mean(),
                'mean_log_prob': all_log_probs.mean(),
                'mean_perplexity': np.exp(-all_log_probs).mean(),
            })
            
            if all_conformal_efficiencies:
                all_conformal_efficiencies = np.concatenate(all_conformal_efficiencies)
                metrics['mean_conformal_efficiency'] = all_conformal_efficiencies.mean()
        
        return metrics
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
