"""Entropy-based uncertainty quantification."""

import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class EntropyUncertainty:
    """
    Uncertainty estimation via Shannon entropy of softmax(logits).
    Higher entropy = more uncertain.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: scaling factor for logits (> 1 = softer probs)
        """
        self.temperature = temperature
    
    @torch.no_grad()
    def get_uncertainty(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute entropy and softmax probabilities.
        
        Args:
            logits: shape (batch, num_classes)
        
        Returns:
            entropy: shape (batch,) - Shannon entropy per sample
            probs: shape (batch, num_classes) - softmax probabilities
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Softmax probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Shannon entropy: H = -sum(p * log(p))
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch,)
        
        return entropy, probs
    
    @torch.no_grad()
    def get_calibrated_confidence(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get confidence score: 1 / (1 + entropy).
        Higher = more confident.
        
        Args:
            logits: shape (batch, num_classes)
        
        Returns:
            confidence: shape (batch,) in [0, 1]
        """
        entropy, _ = self.get_uncertainty(logits)
        # Normalize entropy to [0, 1]: max entropy for K classes is ln(K)
        num_classes = logits.shape[-1]
        max_entropy = np.log(num_classes)
        norm_entropy = entropy / max_entropy
        confidence = 1.0 - norm_entropy
        return confidence
