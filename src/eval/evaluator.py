"""Evaluation pipeline for contrastive learning."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from ..utils.device import get_device, load_checkpoint
from ..metrics.evaluation import compute_all_metrics, ContrastiveMetrics, RetrievalMetrics, ClusteringMetrics


console = Console()


class ContrastiveEvaluator:
    """Evaluator for contrastive learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        k_values: List[int] = [1, 5, 10, 20],
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: Device to use
            k_values: List of k values for k-NN evaluation
        """
        self.model = model
        self.device = device or get_device()
        self.k_values = k_values
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics
        self.contrastive_metrics = ContrastiveMetrics(k_values=k_values)
        self.retrieval_metrics = RetrievalMetrics(k_values=k_values)
        self.clustering_metrics = ClusteringMetrics()
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = load_checkpoint(checkpoint_path, self.model, device=self.device)
        console.print(f"Loaded checkpoint from {checkpoint_path}")
    
    def extract_features(
        self,
        data_loader: DataLoader,
        use_embeddings: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from dataset.
        
        Args:
            data_loader: Data loader
            use_embeddings: Whether to use normalized embeddings
            
        Returns:
            Tuple of (features, labels)
        """
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Extract features
                    if use_embeddings and hasattr(self.model, "get_embeddings"):
                        features = self.model.get_embeddings(images)
                    elif hasattr(self.model, "encode"):
                        features = self.model.encode(images)
                    else:
                        features = self.model(images)
                    
                    # Store features and labels
                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())
        
        # Concatenate all features and labels
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return features, labels
    
    def evaluate_knn(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate using k-NN classification.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            Dictionary of k-NN metrics
        """
        console.print("Extracting training features...")
        train_features, train_labels = self.extract_features(train_loader)
        
        console.print("Extracting test features...")
        test_features, test_labels = self.extract_features(test_loader)
        
        # Compute k-NN metrics
        metrics = self.contrastive_metrics.compute_metrics(
            test_features, test_labels, train_features, train_labels
        )
        
        return metrics
    
    def evaluate_linear(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate using linear classification.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            Dictionary of linear metrics
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report
        
        console.print("Extracting training features...")
        train_features, train_labels = self.extract_features(train_loader)
        
        console.print("Extracting test features...")
        test_features, test_labels = self.extract_features(test_loader)
        
        # Convert to numpy
        train_features_np = train_features.numpy()
        train_labels_np = train_labels.numpy()
        test_features_np = test_features.numpy()
        test_labels_np = test_labels.numpy()
        
        metrics = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(train_features_np, train_labels_np)
        lr_pred = lr.predict(test_features_np)
        lr_accuracy = accuracy_score(test_labels_np, lr_pred)
        metrics["linear_accuracy"] = lr_accuracy
        
        # SVM
        svm = SVC(random_state=42)
        svm.fit(train_features_np, train_labels_np)
        svm_pred = svm.predict(test_features_np)
        svm_accuracy = accuracy_score(test_labels_np, svm_pred)
        metrics["svm_accuracy"] = svm_accuracy
        
        return metrics
    
    def evaluate_retrieval(
        self,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate using retrieval metrics.
        
        Args:
            query_loader: Query data loader
            gallery_loader: Gallery data loader
            
        Returns:
            Dictionary of retrieval metrics
        """
        console.print("Extracting query features...")
        query_features, query_labels = self.extract_features(query_loader)
        
        console.print("Extracting gallery features...")
        gallery_features, gallery_labels = self.extract_features(gallery_loader)
        
        # Compute retrieval metrics
        metrics = self.retrieval_metrics.compute_metrics(
            query_features, query_labels, gallery_features, gallery_labels
        )
        
        return metrics
    
    def evaluate_clustering(
        self,
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate using clustering metrics.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Dictionary of clustering metrics
        """
        console.print("Extracting features...")
        features, labels = self.extract_features(data_loader)
        
        # Compute clustering metrics
        metrics = self.clustering_metrics.compute_metrics(features, labels)
        
        return metrics
    
    def evaluate_all(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        query_loader: Optional[DataLoader] = None,
        gallery_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using all metrics.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            query_loader: Query data loader (for retrieval)
            gallery_loader: Gallery data loader (for retrieval)
            
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # k-NN evaluation
        console.print("Evaluating k-NN classification...")
        knn_metrics = self.evaluate_knn(train_loader, test_loader)
        all_metrics.update(knn_metrics)
        
        # Linear evaluation
        console.print("Evaluating linear classification...")
        linear_metrics = self.evaluate_linear(train_loader, test_loader)
        all_metrics.update(linear_metrics)
        
        # Clustering evaluation
        console.print("Evaluating clustering...")
        clustering_metrics = self.evaluate_clustering(test_loader)
        all_metrics.update(clustering_metrics)
        
        # Retrieval evaluation (if provided)
        if query_loader is not None and gallery_loader is not None:
            console.print("Evaluating retrieval...")
            retrieval_metrics = self.evaluate_retrieval(query_loader, gallery_loader)
            all_metrics.update(retrieval_metrics)
        
        return all_metrics
    
    def print_results(self, metrics: Dict[str, float]) -> None:
        """
        Print evaluation results in a formatted table.
        
        Args:
            metrics: Dictionary of metrics
        """
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in metrics.items():
            table.add_row(key, f"{value:.4f}")
        
        console.print(table)
    
    def save_results(
        self,
        metrics: Dict[str, float],
        save_path: str,
    ) -> None:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save results
        """
        import json
        
        # Convert numpy types to Python types
        metrics_serializable = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                metrics_serializable[key] = value.item()
            else:
                metrics_serializable[key] = value
        
        # Save to JSON
        with open(save_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        
        console.print(f"Results saved to {save_path}")


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    save_results: bool = True,
    results_path: str = "./evaluation_results.json",
) -> Dict[str, float]:
    """
    Evaluate a contrastive learning model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        test_loader: Test data loader
        checkpoint_path: Path to model checkpoint
        device: Device to use
        save_results: Whether to save results
        results_path: Path to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize evaluator
    evaluator = ContrastiveEvaluator(model, device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        evaluator.load_checkpoint(checkpoint_path)
    
    # Evaluate model
    metrics = evaluator.evaluate_all(train_loader, test_loader)
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results if requested
    if save_results:
        evaluator.save_results(metrics, results_path)
    
    return metrics
