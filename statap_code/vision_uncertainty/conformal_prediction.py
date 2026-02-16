"""Conformal Prediction for uncertainty quantification."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ConformalPrediction:
    """
    Conformal prediction for obtaining prediction sets with coverage guarantees.
    Provides calibrated uncertainty intervals.
    """
    
    def __init__(self, confidence_level: float = 0.9):
        """
        Args:
            confidence_level: desired coverage probability (e.g., 0.9 = 90%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.quantile_val = None
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Calibrate on validation set to find the quantile.
        
        Args:
            logits: shape (batch, num_classes)
            labels: shape (batch,) - ground truth labels
        
        Returns:
            quantile_val: the calibrated quantile threshold
        """
        with torch.no_grad():
            # Compute non-conformity scores = -log_prob of true class
            log_probs = F.log_softmax(logits, dim=-1)
            true_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            scores = -true_log_probs  # higher = less confident
            
            # Compute quantile
            n = len(scores)
            quantile_idx = int(np.ceil((n + 1) * (1 - self.alpha)) - 1)
            quantile_idx = min(quantile_idx, n - 1)
            
            sorted_scores, _ = torch.sort(scores)
            self.quantile_val = sorted_scores[quantile_idx].item()
        
        return self.quantile_val
    
    @torch.no_grad()
    def get_prediction_set(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction sets: classes where log-prob is above threshold.
        
        Args:
            logits: shape (batch, num_classes)
        
        Returns:
            pred_sets: shape (batch, num_classes) - boolean mask
            set_sizes: shape (batch,) - size of each prediction set
        """
        if self.quantile_val is None:
            raise ValueError("Must call calibrate() first")
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Include classes where -log_prob <= quantile (i.e., log_prob >= -quantile)
        pred_sets = (-log_probs) <= self.quantile_val  # (batch, num_classes)
        set_sizes = pred_sets.sum(dim=-1)  # (batch,)
        
        return pred_sets, set_sizes
    
    @torch.no_grad()
    def get_efficiency_measure(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Measure of prediction set efficiency (inverse of set size).
        Higher = more confident/efficient.
        
        Args:
            logits: shape (batch, num_classes)
        
        Returns:
            efficiency: shape (batch,) in [0, 1]
            set_sizes: shape (batch,)
        """
        pred_sets, set_sizes = self.get_prediction_set(logits)
        
        num_classes = logits.shape[-1]
        efficiency = 1.0 - (set_sizes.float() / num_classes)
        
        return efficiency, set_sizes
    
    @torch.no_grad()
    def get_coverage(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute empirical coverage on a dataset.
        
        Args:
            logits: shape (batch, num_classes)
            labels: shape (batch,)
        
        Returns:
            coverage: fraction of true labels in prediction sets
        """
        pred_sets, _ = self.get_prediction_set(logits)
        
        # Check if true label is in prediction set
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[-1])
        is_covered = (pred_sets & labels_one_hot.bool()).any(dim=-1)
        
        coverage = is_covered.float().mean().item()
        return coverage
