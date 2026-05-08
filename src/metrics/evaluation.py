"""Evaluation metrics for contrastive learning."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier


class ContrastiveMetrics:
    """Metrics for contrastive learning evaluation."""
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        """
        Initialize metrics.
        
        Args:
            k_values: List of k values for k-NN evaluation
        """
        self.k_values = k_values
    
    def compute_metrics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_features: Optional[torch.Tensor] = None,
        train_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            features: Feature embeddings
            labels: Ground truth labels
            train_features: Training features (for k-NN)
            train_labels: Training labels (for k-NN)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        if train_features is not None and train_labels is not None:
            train_features_np = train_features.cpu().numpy()
            train_labels_np = train_labels.cpu().numpy()
            
            # k-NN evaluation
            knn_metrics = self._compute_knn_metrics(
                train_features_np, train_labels_np, features_np, labels_np
            )
            metrics.update(knn_metrics)
        
        # Linear evaluation (if we have enough data)
        if len(features_np) > 100:
            linear_metrics = self._compute_linear_metrics(features_np, labels_np)
            metrics.update(linear_metrics)
        
        return metrics
    
    def _compute_knn_metrics(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute k-NN metrics."""
        metrics = {}
        
        for k in self.k_values:
            # Fit k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(train_features, train_labels)
            
            # Predict
            predictions = knn.predict(test_features)
            
            # Compute accuracy
            accuracy = accuracy_score(test_labels, predictions)
            metrics[f"knn_{k}_accuracy"] = accuracy
            
            # Compute precision, recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels, predictions, average="weighted"
            )
            metrics[f"knn_{k}_precision"] = precision
            metrics[f"knn_{k}_recall"] = recall
            metrics[f"knn_{k}_f1"] = f1
        
        return metrics
    
    def _compute_linear_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute linear evaluation metrics."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        metrics = {}
        
        # Fit logistic regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Cross-validation
        cv_scores = cross_val_score(lr, features, labels, cv=5, scoring="accuracy")
        metrics["linear_accuracy"] = cv_scores.mean()
        metrics["linear_accuracy_std"] = cv_scores.std()
        
        return metrics


class RetrievalMetrics:
    """Retrieval metrics for contrastive learning."""
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50]):
        """
        Initialize retrieval metrics.
        
        Args:
            k_values: List of k values for retrieval evaluation
        """
        self.k_values = k_values
    
    def compute_metrics(
        self,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Args:
            query_features: Query embeddings
            query_labels: Query labels
            gallery_features: Gallery embeddings
            gallery_labels: Gallery labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Compute similarities
        similarities = torch.matmul(query_features, gallery_features.T)
        
        # Sort by similarity
        _, indices = torch.sort(similarities, dim=-1, descending=True)
        
        # Compute metrics for each k
        for k in self.k_values:
            # Get top-k predictions
            top_k_indices = indices[:, :k]
            top_k_labels = gallery_labels[top_k_indices]
            
            # Compute recall@k
            recall_k = self._compute_recall_at_k(query_labels, top_k_labels)
            metrics[f"recall@{k}"] = recall_k
            
            # Compute precision@k
            precision_k = self._compute_precision_at_k(query_labels, top_k_labels)
            metrics[f"precision@{k}"] = precision_k
        
        # Compute mAP
        map_score = self._compute_map(query_labels, gallery_labels, similarities)
        metrics["mAP"] = map_score
        
        return metrics
    
    def _compute_recall_at_k(
        self,
        query_labels: torch.Tensor,
        top_k_labels: torch.Tensor,
    ) -> float:
        """Compute recall@k."""
        batch_size = query_labels.size(0)
        recalls = []
        
        for i in range(batch_size):
            query_label = query_labels[i]
            top_k_label = top_k_labels[i]
            
            # Count relevant items in top-k
            relevant_count = (top_k_label == query_label).sum().float()
            
            # Count total relevant items
            total_relevant = (top_k_label == query_label).sum().float()
            
            if total_relevant > 0:
                recall = relevant_count / total_relevant
                recalls.append(recall.item())
        
        return np.mean(recalls) if recalls else 0.0
    
    def _compute_precision_at_k(
        self,
        query_labels: torch.Tensor,
        top_k_labels: torch.Tensor,
    ) -> float:
        """Compute precision@k."""
        batch_size = query_labels.size(0)
        precisions = []
        
        for i in range(batch_size):
            query_label = query_labels[i]
            top_k_label = top_k_labels[i]
            
            # Count relevant items in top-k
            relevant_count = (top_k_label == query_label).sum().float()
            
            # Count total items in top-k
            total_count = top_k_label.size(0)
            
            precision = relevant_count / total_count
            precisions.append(precision.item())
        
        return np.mean(precisions)
    
    def _compute_map(
        self,
        query_labels: torch.Tensor,
        gallery_labels: torch.Tensor,
        similarities: torch.Tensor,
    ) -> float:
        """Compute mean Average Precision (mAP)."""
        batch_size = query_labels.size(0)
        aps = []
        
        for i in range(batch_size):
            query_label = query_labels[i]
            sim_scores = similarities[i]
            
            # Sort by similarity
            _, indices = torch.sort(sim_scores, descending=True)
            sorted_labels = gallery_labels[indices]
            
            # Compute AP
            ap = self._compute_ap(query_label, sorted_labels)
            aps.append(ap)
        
        return np.mean(aps)
    
    def _compute_ap(
        self,
        query_label: torch.Tensor,
        sorted_labels: torch.Tensor,
    ) -> float:
        """Compute Average Precision (AP)."""
        relevant_mask = (sorted_labels == query_label).float()
        
        if relevant_mask.sum() == 0:
            return 0.0
        
        # Compute precision at each position
        precisions = []
        for i in range(len(sorted_labels)):
            if relevant_mask[i] == 1:
                precision = relevant_mask[:i+1].sum() / (i + 1)
                precisions.append(precision.item())
        
        return np.mean(precisions) if precisions else 0.0


class ClusteringMetrics:
    """Clustering metrics for contrastive learning."""
    
    def __init__(self):
        """Initialize clustering metrics."""
        pass
    
    def compute_metrics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute clustering metrics.
        
        Args:
            features: Feature embeddings
            labels: Ground truth labels
            n_clusters: Number of clusters (if None, use number of unique labels)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        metrics = {}
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = len(np.unique(labels_np))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_np)
        
        # Compute metrics
        ari = adjusted_rand_score(labels_np, cluster_labels)
        nmi = normalized_mutual_info_score(labels_np, cluster_labels)
        
        metrics["ari"] = ari
        metrics["nmi"] = nmi
        
        return metrics


def compute_all_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_features: Optional[torch.Tensor] = None,
    train_labels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        features: Feature embeddings
        labels: Ground truth labels
        train_features: Training features (for k-NN)
        train_labels: Training labels (for k-NN)
        
    Returns:
        Dictionary of all metrics
    """
    all_metrics = {}
    
    # Contrastive metrics
    contrastive_metrics = ContrastiveMetrics()
    contrastive_results = contrastive_metrics.compute_metrics(
        features, labels, train_features, train_labels
    )
    all_metrics.update(contrastive_results)
    
    # Clustering metrics
    clustering_metrics = ClusteringMetrics()
    clustering_results = clustering_metrics.compute_metrics(features, labels)
    all_metrics.update(clustering_results)
    
    return all_metrics
