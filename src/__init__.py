"""Contrastive Learning Implementation Package."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .models.simclr import SimCLR, ContrastiveLoss, create_simclr_model
from .models.moco import MoCo, MoCoLoss, create_moco_model
from .models.swav import SwAV, SwAVLoss, create_swav_model
from .models.baselines import (
    BaselineCNN,
    BaselineResNet,
    BaselineViT,
    ClassicalBaseline,
    create_baseline_model,
    create_classical_baseline,
)

from .losses.contrastive import (
    NTXentLoss,
    InfoNCELoss,
    TripletLoss,
    SupConLoss,
    BarlowTwinsLoss,
    get_loss_function,
)

from .data.datasets import (
    CIFAR10Dataset,
    CIFAR100Dataset,
    STL10Dataset,
    create_dataloader,
    get_dataset_stats,
)

from .data.augmentations import (
    SimCLRAugmentation,
    MoCoAugmentation,
    SwAVAugmentation,
    get_augmentation,
)

from .metrics.evaluation import (
    ContrastiveMetrics,
    RetrievalMetrics,
    ClusteringMetrics,
    compute_all_metrics,
)

from .metrics.leaderboard import (
    Leaderboard,
    EvaluationRunner,
    create_leaderboard,
    create_evaluation_runner,
)

from .train.trainer import ContrastiveTrainer
from .train.schedulers import (
    CosineAnnealingWarmupRestarts,
    LinearWarmupCosineAnnealing,
    StepLRWithWarmup,
)

from .eval.evaluator import ContrastiveEvaluator, evaluate_model

from .viz.plots import (
    plot_embeddings,
    plot_confusion_matrix,
    plot_training_curves,
    plot_similarity_matrix,
    plot_retrieval_results,
    create_interactive_plot,
    save_visualizations,
)

from .viz.demo import (
    ContrastiveDemo,
    create_demo_app,
    launch_demo,
)

from .utils.device import (
    get_device,
    set_seed,
    count_parameters,
    get_model_size,
    save_checkpoint,
    load_checkpoint,
)

from .utils.safety import (
    SafetyChecker,
    ComplianceManager,
    EthicsFramework,
    create_safety_checker,
    create_compliance_manager,
    create_ethics_framework,
    print_safety_disclaimer,
)

__all__ = [
    # Models
    "SimCLR",
    "ContrastiveLoss",
    "create_simclr_model",
    "MoCo",
    "MoCoLoss",
    "create_moco_model",
    "SwAV",
    "SwAVLoss",
    "create_swav_model",
    "BaselineCNN",
    "BaselineResNet",
    "BaselineViT",
    "ClassicalBaseline",
    "create_baseline_model",
    "create_classical_baseline",
    
    # Losses
    "NTXentLoss",
    "InfoNCELoss",
    "TripletLoss",
    "SupConLoss",
    "BarlowTwinsLoss",
    "get_loss_function",
    
    # Data
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "STL10Dataset",
    "create_dataloader",
    "get_dataset_stats",
    "SimCLRAugmentation",
    "MoCoAugmentation",
    "SwAVAugmentation",
    "get_augmentation",
    
    # Metrics
    "ContrastiveMetrics",
    "RetrievalMetrics",
    "ClusteringMetrics",
    "compute_all_metrics",
    "Leaderboard",
    "EvaluationRunner",
    "create_leaderboard",
    "create_evaluation_runner",
    
    # Training
    "ContrastiveTrainer",
    "CosineAnnealingWarmupRestarts",
    "LinearWarmupCosineAnnealing",
    "StepLRWithWarmup",
    
    # Evaluation
    "ContrastiveEvaluator",
    "evaluate_model",
    
    # Visualization
    "plot_embeddings",
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_similarity_matrix",
    "plot_retrieval_results",
    "create_interactive_plot",
    "save_visualizations",
    "ContrastiveDemo",
    "create_demo_app",
    "launch_demo",
    
    # Utils
    "get_device",
    "set_seed",
    "count_parameters",
    "get_model_size",
    "save_checkpoint",
    "load_checkpoint",
    "SafetyChecker",
    "ComplianceManager",
    "EthicsFramework",
    "create_safety_checker",
    "create_compliance_manager",
    "create_ethics_framework",
    "print_safety_disclaimer",
]
