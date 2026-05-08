"""Training pipeline for contrastive learning."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID

from ..utils.device import get_device, set_seed, save_checkpoint, load_checkpoint
from ..metrics.evaluation import compute_all_metrics


console = Console()


class ContrastiveTrainer:
    """Trainer for contrastive learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "./checkpoints",
        log_every_n_steps: int = 50,
        val_check_interval: float = 1.0,
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            save_dir: Directory to save checkpoints
            log_every_n_steps: Log every n steps
            val_check_interval: Validation check interval
            save_top_k: Number of best checkpoints to save
            monitor: Metric to monitor
            mode: Mode for monitoring ('min' or 'max')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.save_dir = Path(save_dir)
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.checkpoint_paths = []
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        with Progress() as progress:
            task = progress.add_task(
                f"Training Epoch {epoch}", total=num_batches
            )
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # Standard batch format
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Generate augmented views
                    if hasattr(self.model, "forward"):
                        # SimCLR-style models
                        if len(images.shape) == 4:
                            # Single view, need to create two views
                            # This is a simplified version - in practice, you'd use proper augmentation
                            view1, view2 = images, images
                        else:
                            # Already has two views
                            view1, view2 = images[:, 0], images[:, 1]
                        
                        # Forward pass
                        z1 = self.model(view1)
                        z2 = self.model(view2)
                        
                        # Compute loss
                        loss = self.loss_fn(z1, z2)
                    else:
                        # MoCo-style models
                        loss = self.model(images, images)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                self.global_step += 1
                
                # Log progress
                if batch_idx % self.log_every_n_steps == 0:
                    console.print(
                        f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                        f"Loss: {loss.item():.4f}"
                    )
                
                progress.update(task, advance=1)
        
        # Compute average loss
        avg_loss = total_loss / num_batches
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {"train_loss": avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                # Move batch to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    if hasattr(self.model, "encode"):
                        features = self.model.encode(images)
                    else:
                        features = self.model(images)
                    
                    # Store features and labels
                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())
        
        # Concatenate all features and labels
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute evaluation metrics
        metrics = compute_all_metrics(all_features, all_labels)
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            checkpoint = load_checkpoint(
                resume_from_checkpoint, self.model, self.optimizer, self.scheduler
            )
            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self.best_metric = checkpoint.get("best_metric", self.best_metric)
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_metrics = {}
            if val_loader is not None and epoch % int(self.val_check_interval) == 0:
                val_metrics = self.validate(val_loader, epoch)
                self.val_metrics.append(val_metrics)
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(epoch, train_metrics, val_metrics)
        
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Log metrics to console."""
        table = Table(title=f"Epoch {epoch} Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Add training metrics
        for key, value in train_metrics.items():
            table.add_row(key, f"{value:.4f}")
        
        # Add validation metrics
        for key, value in val_metrics.items():
            table.add_row(key, f"{value:.4f}")
        
        console.print(table)
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Save checkpoint if it's one of the best."""
        if not val_metrics:
            return
        
        # Get metric value
        metric_value = val_metrics.get(self.monitor, float("inf"))
        
        # Check if this is a better checkpoint
        is_better = (
            (self.mode == "min" and metric_value < self.best_metric) or
            (self.mode == "max" and metric_value > self.best_metric)
        )
        
        if is_better:
            self.best_metric = metric_value
            
            # Save checkpoint
            checkpoint_path = self.save_dir / f"best_model_epoch_{epoch}.pth"
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                train_metrics.get("train_loss", 0.0),
                str(checkpoint_path),
                global_step=self.global_step,
                best_metric=self.best_metric,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            
            # Track checkpoint paths
            self.checkpoint_paths.append(str(checkpoint_path))
            
            # Remove old checkpoints if we exceed save_top_k
            if len(self.checkpoint_paths) > self.save_top_k:
                old_checkpoint = self.checkpoint_paths.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
        
        # Save latest checkpoint
        latest_checkpoint_path = self.save_dir / "latest_model.pth"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            train_metrics.get("train_loss", 0.0),
            str(latest_checkpoint_path),
            global_step=self.global_step,
            best_metric=self.best_metric,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
