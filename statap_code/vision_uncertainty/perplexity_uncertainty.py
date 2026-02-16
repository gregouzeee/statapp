"""Perplexity-based uncertainty quantification (log-prob)."""

import torch
import torch.nn.functional as F
from typing import Tuple
import math


class PerplexityUncertainty:
    """
    Uncertainty estimation via log-probability (negative log softmax).
    Low log-prob (high negative) = more uncertain.
    High log-prob (less negative) = more confident.
    
    Adapted from LLM-based perplexity to vision models.
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
        labels: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-prob and perplexity.
        
        Args:
            logits: shape (batch, num_classes)
            labels: shape (batch,) - ground truth class indices (optional)
        
        Returns:
            log_prob: shape (batch,) - log probability of predicted class
            perplexity: shape (batch,) - exp(-log_prob)
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Log softmax
        log_probs = F.log_softmax(scaled_logits, dim=-1)  # (batch, num_classes)
        
        if labels is not None:
            # Log-prob of ground truth class
            log_prob = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        else:
            # Log-prob of predicted (max) class
            pred_classes = scaled_logits.argmax(dim=-1)
            log_prob = log_probs.gather(1, pred_classes.unsqueeze(1)).squeeze(1)
        
        # Perplexity: exp(-log_prob)
        perplexity = torch.exp(-log_prob)
        
        return log_prob, perplexity
    
    @torch.no_grad()
    def get_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get confidence score based on log-prob.
        Uses sigmoid to map to [0, 1].
        
        Args:
            logits: shape (batch, num_classes)
        
        Returns:
            confidence: shape (batch,) in [0, 1]
        """
        log_prob, _ = self.get_uncertainty(logits)
        
        # Sigmoid of log-prob scaled: higher log_prob -> higher confidence
        # Typical log-probs range from -5 to 0, so scale by 5
        confidence = torch.sigmoid(log_prob * 5.0)
        return confidence
    
    @torch.no_grad()
    def get_avg_logprob_ensemble(
        self,
        logits_list: list,
    ) -> torch.Tensor:
        """
        Compute average log-prob across ensemble members.
        Useful for aggregating predictions.
        
        Args:
            logits_list: list of logits tensors, each shape (batch, num_classes)
        
        Returns:
            avg_logprob: shape (batch,)
        """
        logprobs = []
        for logits in logits_list:
            log_prob, _ = self.get_uncertainty(logits)
            logprobs.append(log_prob)
        
        avg_logprob = torch.stack(logprobs, dim=1).mean(dim=1)  # (batch,)
        return avg_logprob
