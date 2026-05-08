"""Command-line interface for contrastive learning."""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from omegaconf import OmegaConf

from .train.trainer import ContrastiveTrainer
from .eval.evaluator import ContrastiveEvaluator
from .models.simclr import create_simclr_model
from .models.moco import create_moco_model
from .models.swav import create_swav_model
from .models.baselines import create_baseline_model, create_classical_baseline
from .data.datasets import CIFAR10Dataset, CIFAR100Dataset, STL10Dataset, create_dataloader
from .data.augmentations import get_augmentation
from .utils.device import get_device, set_seed
from .viz.demo import create_demo_app


app = typer.Typer(help="Contrastive Learning Implementation")
console = Console()


@app.command()
def train(
    config_path: str = typer.Option(
        "configs/config.yaml",
        help="Path to configuration file"
    ),
    model_type: str = typer.Option(
        "simclr",
        help="Type of model to train (simclr, moco, swav)"
    ),
    dataset: str = typer.Option(
        "cifar10",
        help="Dataset to use (cifar10, cifar100, stl10)"
    ),
    epochs: int = typer.Option(
        100,
        help="Number of epochs to train"
    ),
    batch_size: int = typer.Option(
        256,
        help="Batch size"
    ),
    lr: float = typer.Option(
        0.0001,
        help="Learning rate"
    ),
    device: str = typer.Option(
        "auto",
        help="Device to use (auto, cuda, mps, cpu)"
    ),
    seed: int = typer.Option(
        42,
        help="Random seed"
    ),
    resume: Optional[str] = typer.Option(
        None,
        help="Path to checkpoint to resume from"
    ),
):
    """Train a contrastive learning model."""
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Override config with command line arguments
    config.train.epochs = epochs
    config.train.batch_size = batch_size
    config.optimizer.lr = lr
    config.device = device
    config.seed = seed
    
    # Set seed
    set_seed(seed, config.get("deterministic", True))
    
    # Get device
    device = get_device(config.device)
    
    # Create dataset
    if dataset.lower() == "cifar10":
        train_dataset = CIFAR10Dataset(
            root_dir=config.data.root_dir,
            train=True,
            download=True,
            image_size=config.data.image_size,
        )
    elif dataset.lower() == "cifar100":
        train_dataset = CIFAR100Dataset(
            root_dir=config.data.root_dir,
            train=True,
            download=True,
            image_size=config.data.image_size,
        )
    elif dataset.lower() == "stl10":
        train_dataset = STL10Dataset(
            root_dir=config.data.root_dir,
            split="train",
            download=True,
            image_size=config.data.image_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create augmentation
    augmentation = get_augmentation(
        config.augmentation._target_.split(".")[-1].replace("Augmentation", "").lower(),
        **config.augmentation
    )
    
    # Create data loader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
    )
    
    # Create model
    if model_type.lower() == "simclr":
        model, loss_fn = create_simclr_model(**config.model)
    elif model_type.lower() == "moco":
        model, loss_fn = create_moco_model(**config.model)
    elif model_type.lower() == "swav":
        model, loss_fn = create_swav_model(**config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer
    optimizer = OmegaConf.to_object(config.optimizer)
    optimizer = optimizer["_target_"](model.parameters(), **{k: v for k, v in optimizer.items() if k != "_target_"})
    
    # Create scheduler
    scheduler = None
    if config.get("scheduler"):
        scheduler_config = OmegaConf.to_object(config.scheduler)
        scheduler = scheduler_config["_target_"](optimizer, **{k: v for k, v in scheduler_config.items() if k != "_target_"})
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config.paths.checkpoint_dir,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.logging.val_check_interval,
        save_top_k=config.logging.save_top_k,
        monitor=config.logging.monitor,
        mode=config.logging.mode,
    )
    
    # Train model
    console.print(f"Training {model_type.upper()} model on {dataset.upper()}")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=None,  # No validation for now
        epochs=config.train.epochs,
        resume_from_checkpoint=resume,
    )
    
    console.print("Training completed!")


@app.command()
def eval(
    checkpoint_path: str = typer.Option(
        ...,
        help="Path to model checkpoint"
    ),
    model_type: str = typer.Option(
        "simclr",
        help="Type of model to evaluate (simclr, moco, swav)"
    ),
    dataset: str = typer.Option(
        "cifar10",
        help="Dataset to use (cifar10, cifar100, stl10)"
    ),
    batch_size: int = typer.Option(
        256,
        help="Batch size"
    ),
    device: str = typer.Option(
        "auto",
        help="Device to use (auto, cuda, mps, cpu)"
    ),
    save_results: bool = typer.Option(
        True,
        help="Whether to save results"
    ),
):
    """Evaluate a contrastive learning model."""
    
    # Get device
    device = get_device(device)
    
    # Create dataset
    if dataset.lower() == "cifar10":
        train_dataset = CIFAR10Dataset(
            root_dir="./data",
            train=True,
            download=True,
        )
        test_dataset = CIFAR10Dataset(
            root_dir="./data",
            train=False,
            download=True,
        )
    elif dataset.lower() == "cifar100":
        train_dataset = CIFAR100Dataset(
            root_dir="./data",
            train=True,
            download=True,
        )
        test_dataset = CIFAR100Dataset(
            root_dir="./data",
            train=False,
            download=True,
        )
    elif dataset.lower() == "stl10":
        train_dataset = STL10Dataset(
            root_dir="./data",
            split="train",
            download=True,
        )
        test_dataset = STL10Dataset(
            root_dir="./data",
            split="test",
            download=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Create model
    if model_type.lower() == "simclr":
        model, _ = create_simclr_model()
    elif model_type.lower() == "moco":
        model, _ = create_moco_model()
    elif model_type.lower() == "swav":
        model, _ = create_swav_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create evaluator
    evaluator = ContrastiveEvaluator(model, device)
    
    # Load checkpoint
    evaluator.load_checkpoint(checkpoint_path)
    
    # Evaluate model
    metrics = evaluator.evaluate_all(train_loader, test_loader)
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results if requested
    if save_results:
        results_path = f"evaluation_results_{model_type}_{dataset}.json"
        evaluator.save_results(metrics, results_path)


@app.command()
def demo(
    checkpoint_path: Optional[str] = typer.Option(
        None,
        help="Path to model checkpoint"
    ),
    model_type: str = typer.Option(
        "simclr",
        help="Type of model to use (simclr, moco, swav)"
    ),
    port: int = typer.Option(
        8501,
        help="Port for the demo app"
    ),
):
    """Launch interactive demo."""
    
    # Create demo app
    demo_app = create_demo_app(model_type, checkpoint_path)
    
    # Launch app
    console.print(f"Launching demo on port {port}")
    demo_app.launch(server_port=port)


@app.command()
def compare_baselines(
    dataset: str = typer.Option(
        "cifar10",
        help="Dataset to use (cifar10, cifar100, stl10)"
    ),
    batch_size: int = typer.Option(
        256,
        help="Batch size"
    ),
    device: str = typer.Option(
        "auto",
        help="Device to use (auto, cuda, mps, cpu)"
    ),
):
    """Compare different baseline methods."""
    
    # Get device
    device = get_device(device)
    
    # Create dataset
    if dataset.lower() == "cifar10":
        train_dataset = CIFAR10Dataset(
            root_dir="./data",
            train=True,
            download=True,
        )
        test_dataset = CIFAR10Dataset(
            root_dir="./data",
            train=False,
            download=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Extract features for classical baselines
    console.print("Extracting features for classical baselines...")
    
    # Use a simple CNN to extract features
    feature_extractor = create_baseline_model("cnn", num_classes=10)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    with torch.no_grad():
        for batch in train_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            features = feature_extractor(images)
            train_features.append(features.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
        
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            features = feature_extractor(images)
            test_features.append(features.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    
    # Concatenate features
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Compare classical baselines
    methods = ["random_forest", "logistic", "svm", "knn"]
    results = {}
    
    for method in methods:
        console.print(f"Training {method}...")
        
        baseline = create_classical_baseline(method)
        baseline.fit(train_features, train_labels)
        
        predictions = baseline.predict(test_features)
        accuracy = (predictions == test_labels).mean()
        
        results[method] = accuracy
    
    # Print results
    table = Table(title="Baseline Comparison Results")
    table.add_column("Method", style="cyan")
    table.add_column("Accuracy", style="magenta")
    
    for method, accuracy in results.items():
        table.add_row(method, f"{accuracy:.4f}")
    
    console.print(table)


if __name__ == "__main__":
    app()
