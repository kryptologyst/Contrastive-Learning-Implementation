"""Leaderboard and evaluation results management."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from .evaluation import compute_all_metrics


console = Console()


class Leaderboard:
    """Leaderboard for tracking model performance."""
    
    def __init__(self, results_file: str = "./assets/results/leaderboard.json"):
        """
        Initialize leaderboard.
        
        Args:
            results_file: Path to results file
        """
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing results
        self.results = self._load_results()
    
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load existing results from file."""
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                return json.load(f)
        return []
    
    def _save_results(self) -> None:
        """Save results to file."""
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def add_result(
        self,
        model_name: str,
        dataset: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Add a new result to the leaderboard.
        
        Args:
            model_name: Name of the model
            dataset: Dataset name
            metrics: Evaluation metrics
            config: Model configuration
            notes: Additional notes
        """
        result = {
            "model_name": model_name,
            "dataset": dataset,
            "metrics": metrics,
            "config": config or {},
            "notes": notes or "",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        self.results.append(result)
        self._save_results()
        
        console.print(f"✅ Added result for {model_name} on {dataset}")
    
    def get_results(
        self,
        model_name: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get results filtered by model and dataset.
        
        Args:
            model_name: Filter by model name
            dataset: Filter by dataset
            
        Returns:
            List of filtered results
        """
        filtered_results = self.results
        
        if model_name:
            filtered_results = [r for r in filtered_results if r["model_name"] == model_name]
        
        if dataset:
            filtered_results = [r for r in filtered_results if r["dataset"] == dataset]
        
        return filtered_results
    
    def get_best_results(
        self,
        metric: str = "knn_1_accuracy",
        dataset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get best results for a specific metric.
        
        Args:
            metric: Metric to rank by
            dataset: Filter by dataset
            
        Returns:
            List of best results
        """
        results = self.get_results(dataset=dataset)
        
        # Sort by metric value
        results.sort(
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True
        )
        
        return results
    
    def print_leaderboard(
        self,
        dataset: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Print leaderboard table.
        
        Args:
            dataset: Filter by dataset
            metrics: List of metrics to display
        """
        results = self.get_results(dataset=dataset)
        
        if not results:
            console.print("No results found.")
            return
        
        # Default metrics to display
        if metrics is None:
            metrics = [
                "knn_1_accuracy",
                "knn_5_accuracy",
                "linear_accuracy",
                "ari",
                "nmi",
            ]
        
        # Create table
        table = Table(title=f"Leaderboard - {dataset or 'All Datasets'}")
        table.add_column("Rank", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Dataset", style="green")
        
        # Add metric columns
        for metric in metrics:
            table.add_column(metric.replace("_", " ").title(), style="yellow")
        
        # Add results
        for i, result in enumerate(results, 1):
            row = [
                str(i),
                result["model_name"],
                result["dataset"],
            ]
            
            for metric in metrics:
                value = result["metrics"].get(metric, 0.0)
                row.append(f"{value:.4f}")
            
            table.add_row(*row)
        
        console.print(table)
    
    def export_to_csv(self, output_path: str) -> None:
        """
        Export results to CSV.
        
        Args:
            output_path: Path to save CSV
        """
        if not self.results:
            console.print("No results to export.")
            return
        
        # Flatten results for CSV
        flattened_results = []
        for result in self.results:
            flat_result = {
                "model_name": result["model_name"],
                "dataset": result["dataset"],
                "timestamp": result["timestamp"],
                "notes": result["notes"],
            }
            
            # Add metrics
            for key, value in result["metrics"].items():
                flat_result[f"metric_{key}"] = value
            
            # Add config
            for key, value in result["config"].items():
                flat_result[f"config_{key}"] = value
            
            flattened_results.append(flat_result)
        
        # Create DataFrame and save
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False)
        
        console.print(f"📊 Results exported to {output_path}")
    
    def compare_models(
        self,
        model_names: List[str],
        dataset: Optional[str] = None,
        metric: str = "knn_1_accuracy",
    ) -> None:
        """
        Compare specific models.
        
        Args:
            model_names: List of model names to compare
            dataset: Filter by dataset
            metric: Metric to compare
        """
        results = self.get_results(dataset=dataset)
        
        # Filter by model names
        filtered_results = [r for r in results if r["model_name"] in model_names]
        
        if not filtered_results:
            console.print("No results found for the specified models.")
            return
        
        # Create comparison table
        table = Table(title=f"Model Comparison - {metric}")
        table.add_column("Model", style="cyan")
        table.add_column("Dataset", style="green")
        table.add_column(metric.replace("_", " ").title(), style="magenta")
        table.add_column("Notes", style="yellow")
        
        for result in filtered_results:
            value = result["metrics"].get(metric, 0.0)
            table.add_row(
                result["model_name"],
                result["dataset"],
                f"{value:.4f}",
                result["notes"]
            )
        
        console.print(table)


class EvaluationRunner:
    """Runner for comprehensive model evaluation."""
    
    def __init__(
        self,
        leaderboard: Optional[Leaderboard] = None,
        output_dir: str = "./assets/results",
    ):
        """
        Initialize evaluation runner.
        
        Args:
            leaderboard: Leaderboard instance
            output_dir: Output directory for results
        """
        self.leaderboard = leaderboard or Leaderboard()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a model and add to leaderboard.
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            train_loader: Training data loader
            test_loader: Test data loader
            dataset_name: Name of the dataset
            config: Model configuration
            notes: Additional notes
            
        Returns:
            Evaluation metrics
        """
        console.print(f"🔍 Evaluating {model_name} on {dataset_name}")
        
        # Extract features
        train_features, train_labels = self._extract_features(model, train_loader)
        test_features, test_labels = self._extract_features(model, test_loader)
        
        # Compute metrics
        metrics = compute_all_metrics(
            test_features, test_labels, train_features, train_labels
        )
        
        # Add to leaderboard
        self.leaderboard.add_result(
            model_name=model_name,
            dataset=dataset_name,
            metrics=metrics,
            config=config,
            notes=notes,
        )
        
        # Save detailed results
        self._save_detailed_results(
            model_name, dataset_name, metrics, config, notes
        )
        
        return metrics
    
    def _extract_features(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from model."""
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, labels = batch
                    
                    # Extract features
                    if hasattr(model, "encode"):
                        features = model.encode(images)
                    else:
                        features = model(images)
                    
                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())
        
        # Concatenate
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return features, labels
    
    def _save_detailed_results(
        self,
        model_name: str,
        dataset_name: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]],
        notes: Optional[str],
    ) -> None:
        """Save detailed results to file."""
        result = {
            "model_name": model_name,
            "dataset": dataset_name,
            "metrics": metrics,
            "config": config or {},
            "notes": notes or "",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        # Save individual result
        result_file = self.output_dir / f"{model_name}_{dataset_name}_results.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        
        console.print(f"💾 Detailed results saved to {result_file}")
    
    def run_comprehensive_evaluation(
        self,
        models: Dict[str, torch.nn.Module],
        datasets: Dict[str, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Run comprehensive evaluation on multiple models and datasets.
        
        Args:
            models: Dictionary of model names to models
            datasets: Dictionary of dataset names to (train_loader, test_loader)
            configs: Optional model configurations
        """
        console.print("🚀 Starting comprehensive evaluation")
        
        for model_name, model in models.items():
            for dataset_name, (train_loader, test_loader) in datasets.items():
                config = configs.get(model_name) if configs else None
                
                self.evaluate_model(
                    model=model,
                    model_name=model_name,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    dataset_name=dataset_name,
                    config=config,
                )
        
        # Print final leaderboard
        console.print("\n📊 Final Leaderboard:")
        self.leaderboard.print_leaderboard()
        
        # Export results
        csv_path = self.output_dir / "comprehensive_results.csv"
        self.leaderboard.export_to_csv(str(csv_path))


def create_leaderboard() -> Leaderboard:
    """Create a new leaderboard instance."""
    return Leaderboard()


def create_evaluation_runner(
    leaderboard: Optional[Leaderboard] = None,
    output_dir: str = "./assets/results",
) -> EvaluationRunner:
    """Create a new evaluation runner instance."""
    return EvaluationRunner(leaderboard, output_dir)
