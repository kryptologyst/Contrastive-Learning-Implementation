"""Visualization utilities for contrastive learning."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from ..utils.device import get_device


def plot_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    title: str = "Embedding Visualization",
) -> None:
    """
    Plot embeddings using t-SNE or PCA.
    
    Args:
        embeddings: Embeddings tensor
        labels: Labels tensor
        method: Method to use ('tsne' or 'pca')
        n_components: Number of components
        perplexity: Perplexity for t-SNE
        save_path: Path to save plot
        title: Plot title
    """
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels_np,
            cmap="tab10",
            alpha=0.7,
            s=50,
        )
        plt.colorbar(scatter)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=labels_np,
            cmap="tab10",
            alpha=0.7,
            s=50,
        )
        plt.colorbar(scatter)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")
    
    plt.title(title)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save plot
        title: Plot title
    """
    # Convert to numpy
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> None:
    """
    Plot training curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_metrics: Training metrics
        val_metrics: Validation metrics
        save_path: Path to save plot
        title: Plot title
    """
    # Create subplots
    n_plots = 1
    if val_losses is not None:
        n_plots += 1
    if train_metrics is not None:
        n_plots += len(train_metrics)
    if val_metrics is not None:
        n_plots += len(val_metrics)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot training loss
    axes[plot_idx].plot(train_losses, label="Training Loss", color="blue")
    if val_losses is not None:
        axes[plot_idx].plot(val_losses, label="Validation Loss", color="red")
    axes[plot_idx].set_title("Loss")
    axes[plot_idx].set_xlabel("Epoch")
    axes[plot_idx].set_ylabel("Loss")
    axes[plot_idx].legend()
    axes[plot_idx].grid(True)
    plot_idx += 1
    
    # Plot training metrics
    if train_metrics is not None:
        for metric_name, values in train_metrics.items():
            axes[plot_idx].plot(values, label=f"Training {metric_name}", color="blue")
            if val_metrics is not None and metric_name in val_metrics:
                axes[plot_idx].plot(val_metrics[metric_name], label=f"Validation {metric_name}", color="red")
            axes[plot_idx].set_title(metric_name.title())
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel(metric_name.title())
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_similarity_matrix(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Similarity Matrix",
) -> None:
    """
    Plot similarity matrix of embeddings.
    
    Args:
        embeddings: Embeddings tensor
        labels: Labels tensor (optional)
        save_path: Path to save plot
        title: Plot title
    """
    # Compute similarity matrix
    embeddings_norm = F.normalize(embeddings, dim=-1)
    similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
    
    # Convert to numpy
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # Sort by labels
        labels_np = labels.cpu().numpy()
        sorted_indices = np.argsort(labels_np)
        similarity_matrix_np = similarity_matrix_np[sorted_indices][:, sorted_indices]
        
        # Add class boundaries
        unique_labels = np.unique(labels_np)
        boundaries = []
        for label in unique_labels:
            boundary = np.where(labels_np[sorted_indices] == label)[0]
            if len(boundary) > 0:
                boundaries.append(boundary[-1])
        
        # Plot similarity matrix
        im = plt.imshow(similarity_matrix_np, cmap="viridis", aspect="auto")
        
        # Add class boundaries
        for boundary in boundaries[:-1]:
            plt.axhline(y=boundary + 0.5, color="red", linewidth=2)
            plt.axvline(x=boundary + 0.5, color="red", linewidth=2)
    else:
        im = plt.imshow(similarity_matrix_np, cmap="viridis", aspect="auto")
    
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_retrieval_results(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    top_k: int = 10,
    save_path: Optional[str] = None,
    title: str = "Retrieval Results",
) -> None:
    """
    Plot retrieval results.
    
    Args:
        query_features: Query embeddings
        query_labels: Query labels
        gallery_features: Gallery embeddings
        gallery_labels: Gallery labels
        top_k: Number of top results to show
        save_path: Path to save plot
        title: Plot title
    """
    # Compute similarities
    similarities = torch.matmul(query_features, gallery_features.T)
    
    # Get top-k results
    _, top_k_indices = torch.topk(similarities, top_k, dim=-1)
    
    # Convert to numpy
    query_labels_np = query_labels.cpu().numpy()
    gallery_labels_np = gallery_labels.cpu().numpy()
    top_k_indices_np = top_k_indices.cpu().numpy()
    
    # Create plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(10, len(query_labels_np))):
        query_label = query_labels_np[i]
        top_k_labels = gallery_labels_np[top_k_indices_np[i]]
        
        # Count correct retrievals
        correct_count = (top_k_labels == query_label).sum()
        
        # Plot bar chart
        unique_labels, counts = np.unique(top_k_labels, return_counts=True)
        axes[i].bar(unique_labels, counts)
        axes[i].set_title(f"Query {i} (Label: {query_label})\nCorrect: {correct_count}/{top_k}")
        axes[i].set_xlabel("Retrieved Label")
        axes[i].set_ylabel("Count")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def create_interactive_plot(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    title: str = "Interactive Embedding Visualization",
) -> None:
    """
    Create interactive plot using Plotly.
    
    Args:
        embeddings: Embeddings tensor
        labels: Labels tensor
        method: Method to use ('tsne' or 'pca')
        n_components: Number of components
        perplexity: Perplexity for t-SNE
        save_path: Path to save plot
        title: Plot title
    """
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings_np)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create interactive plot
    if n_components == 2:
        fig = px.scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            color=labels_np,
            title=title,
            labels={"x": f"{method.upper()} Component 1", "y": f"{method.upper()} Component 2"},
        )
    elif n_components == 3:
        fig = px.scatter_3d(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            z=reduced_embeddings[:, 2],
            color=labels_np,
            title=title,
            labels={"x": f"{method.upper()} Component 1", "y": f"{method.upper()} Component 2", "z": f"{method.upper()} Component 3"},
        )
    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
    )
    
    # Show plot
    fig.show()
    
    # Save plot
    if save_path:
        fig.write_html(save_path)


def save_visualizations(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str = "./assets/plots",
    prefix: str = "contrastive",
) -> None:
    """
    Save all visualizations.
    
    Args:
        embeddings: Embeddings tensor
        labels: Labels tensor
        output_dir: Output directory
        prefix: Prefix for filenames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save t-SNE plot
    plot_embeddings(
        embeddings,
        labels,
        method="tsne",
        save_path=str(output_path / f"{prefix}_tsne.png"),
        title=f"{prefix.title()} t-SNE Visualization",
    )
    
    # Save PCA plot
    plot_embeddings(
        embeddings,
        labels,
        method="pca",
        save_path=str(output_path / f"{prefix}_pca.png"),
        title=f"{prefix.title()} PCA Visualization",
    )
    
    # Save similarity matrix
    plot_similarity_matrix(
        embeddings,
        labels,
        save_path=str(output_path / f"{prefix}_similarity.png"),
        title=f"{prefix.title()} Similarity Matrix",
    )
    
    # Save interactive plot
    create_interactive_plot(
        embeddings,
        labels,
        method="tsne",
        save_path=str(output_path / f"{prefix}_interactive.html"),
        title=f"{prefix.title()} Interactive Visualization",
    )
