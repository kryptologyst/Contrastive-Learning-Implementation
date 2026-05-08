"""Loss functions for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) loss."""
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: First view embeddings
            z_j: Second view embeddings
            
        Returns:
            NT-Xent loss
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        labels = labels[mask]
        
        # Remove diagonal elements
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings
            positive: Positive embeddings
            negatives: Negative embeddings (optional)
            
        Returns:
            InfoNCE loss
        """
        batch_size = query.size(0)
        
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # Positive logits
        pos_logits = torch.sum(query * positive, dim=-1) / self.temperature
        
        if negatives is not None:
            # Normalize negatives
            negatives = F.normalize(negatives, dim=-1)
            
            # Negative logits
            neg_logits = torch.matmul(query, negatives.T) / self.temperature
            
            # Concatenate logits
            logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        else:
            # Use all other samples in batch as negatives
            all_embeddings = torch.cat([query, positive], dim=0)
            all_logits = torch.matmul(query, all_embeddings.T) / self.temperature
            
            # Remove self-similarity
            logits = all_logits[torch.arange(batch_size), batch_size:]
        
        # Labels (positive pairs are at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long).to(query.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for contrastive learning."""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin parameter
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Triplet loss
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Compute loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings
            labels: Class labels
            
        Returns:
            Supervised contrastive loss
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements
        mask = mask - torch.eye(batch_size).to(features.device)
        
        # Compute log probabilities
        log_prob = F.log_softmax(similarity_matrix, dim=-1)
        
        # Compute loss for each sample
        loss = -torch.sum(mask * log_prob, dim=-1) / torch.sum(mask, dim=-1)
        
        return loss.mean()


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss."""
    
    def __init__(self, lambda_param: float = 5e-3):
        """
        Initialize Barlow Twins loss.
        
        Args:
            lambda_param: Lambda parameter
        """
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Barlow Twins loss.
        
        Args:
            z1: First view embeddings
            z2: Second view embeddings
            
        Returns:
            Barlow Twins loss
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Compute cross-correlation matrix
        c = torch.matmul(z1.T, z2) / batch_size
        
        # Compute loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = c.flatten()[1:].view(c.size(0) - 1, c.size(0) + 1)[:, ::c.size(0) + 1].flatten().pow_(2).sum()
        
        loss = on_diag + self.lambda_param * off_diag
        
        return loss


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Get loss function by type.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Loss function parameters
        
    Returns:
        Loss function
    """
    if loss_type.lower() == "ntxent":
        return NTXentLoss(**kwargs)
    elif loss_type.lower() == "infonce":
        return InfoNCELoss(**kwargs)
    elif loss_type.lower() == "triplet":
        return TripletLoss(**kwargs)
    elif loss_type.lower() == "supcon":
        return SupConLoss(**kwargs)
    elif loss_type.lower() == "barlow":
        return BarlowTwinsLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
