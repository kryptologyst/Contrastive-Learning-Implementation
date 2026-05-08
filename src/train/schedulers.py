"""Learning rate schedulers for contrastive learning."""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with warmup and restarts."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            T_0: Initial restart period
            T_mult: Factor to increase T_0 after each restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.T_cur = 0
        self.T_i = T_0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        """Get learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            if self.last_epoch >= self.T_cur + self.T_i:
                self.T_cur = self.last_epoch
                self.T_i *= self.T_mult
            
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.T_cur) / self.T_i)) / 2
                for base_lr in self.base_lrs
            ]


class LinearWarmupCosineAnnealing(_LRScheduler):
    """Linear warmup followed by cosine annealing."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of epochs
            eta_min: Minimum learning rate
            last_epoch: Last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        """Get learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_epochs = self.max_epochs - self.warmup_epochs
            
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * cosine_epoch / cosine_epochs)) / 2
                for base_lr in self.base_lrs
            ]


class StepLRWithWarmup(_LRScheduler):
    """Step LR with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            step_size: Period of learning rate decay
            gamma: Multiplicative factor of learning rate decay
            warmup_epochs: Number of warmup epochs
            last_epoch: Last epoch
        """
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        """Get learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Step decay phase
            return [
                base_lr * self.gamma ** ((self.last_epoch - self.warmup_epochs) // self.step_size)
                for base_lr in self.base_lrs
            ]
