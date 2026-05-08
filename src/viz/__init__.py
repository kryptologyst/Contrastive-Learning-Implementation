"""Visualization module."""

from .plots import (
    plot_embeddings,
    plot_confusion_matrix,
    plot_training_curves,
    plot_similarity_matrix,
    plot_retrieval_results,
    create_interactive_plot,
    save_visualizations,
)

from .demo import (
    ContrastiveDemo,
    create_demo_app,
    launch_demo,
)

__all__ = [
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
]
