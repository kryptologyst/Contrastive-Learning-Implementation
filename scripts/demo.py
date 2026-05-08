#!/usr/bin/env python3
"""Demo script for contrastive learning implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models.simclr import create_simclr_model
from src.data.datasets import CIFAR10Dataset
from src.data.augmentations import SimCLRAugmentation
from src.utils.device import get_device, set_seed
from src.utils.safety import print_safety_disclaimer
from src.viz.plots import plot_embeddings


def main():
    """Main demo function."""
    print("🚀 Contrastive Learning Implementation Demo")
    print("=" * 50)
    
    # Print safety disclaimer
    print_safety_disclaimer()
    
    # Set seed for reproducibility
    set_seed(42, deterministic=True)
    
    # Get device
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Create model
    print("\n📦 Creating SimCLR model...")
    model, loss_fn = create_simclr_model(
        base_model="resnet18",
        pretrained=False,
        projection_dim=128,
        temperature=0.5,
    )
    model.to(device)
    
    # Create dataset
    print("📊 Loading CIFAR-10 dataset...")
    dataset = CIFAR10Dataset(
        root_dir="./data",
        train=True,
        download=True,
        image_size=224,
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    
    # Create augmentation
    augmentation = SimCLRAugmentation(image_size=224)
    
    # Demo training step
    print("\n🏋️ Demo training step...")
    model.train()
    
    # Get a batch
    batch = next(iter(dataloader))
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    
    # Create augmented views (simplified)
    view1, view2 = images, images  # In practice, use proper augmentation
    
    # Forward pass
    z1 = model(view1)
    z2 = model(view2)
    
    # Compute loss
    loss = loss_fn(z1, z2)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Demo evaluation
    print("\n🔍 Demo evaluation...")
    model.eval()
    
    # Extract features
    with torch.no_grad():
        features = model.encode(images)
        features_norm = torch.nn.functional.normalize(features, dim=-1)
    
    print(f"Features shape: {features.shape}")
    print(f"Features norm: {torch.norm(features_norm, dim=-1).mean():.4f}")
    
    # Demo visualization (if matplotlib is available)
    try:
        print("\n📈 Creating embedding visualization...")
        plot_embeddings(
            features[:100],  # Use first 100 samples
            labels[:100],
            method="pca",
            title="SimCLR Embeddings (PCA)",
        )
        print("✅ Visualization created successfully!")
    except ImportError:
        print("⚠️ Matplotlib not available, skipping visualization")
    
    # Model statistics
    print("\n📊 Model Statistics:")
    from src.utils.device import count_parameters, get_model_size
    param_count = count_parameters(model)
    model_size = get_model_size(model)
    
    print(f"Parameters: {param_count:,}")
    print(f"Model size: {model_size}")
    
    print("\n✅ Demo completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Train the model: python -m src.cli train --model-type simclr")
    print("2. Evaluate: python -m src.cli eval --checkpoint-path <path>")
    print("3. Launch demo: python -m src.cli demo --model-type simclr")
    print("4. Compare baselines: python -m src.cli compare-baselines")


if __name__ == "__main__":
    main()
